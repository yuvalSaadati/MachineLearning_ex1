# Yuval Saadaty 205956634
import sys
import numpy as np
import scipy
from scipy.io import wavfile


def compress_file(sample, centroids, output):
    # samplerate - sampling rate in samples per second
    # values - a numpy array with all the values read from the file
    samplerate, values = scipy.io.wavfile.read(sample)
    # wav_values are all wav file values
    wav_values = np.array(values.copy())
    # numpy array that contain all wav values with new value which is the value of the closet centroid to it
    new_values = np.empty(shape=wav_values.shape, dtype='int16')
    centroids = np.loadtxt(centroids)
    # add 3 columns to save count and amount of values (x,y) that relate to each centroid
    ones_col = np.zeros((centroids.shape[0],3))
    centroids = np.hstack((centroids, ones_col))
    # run for 30 iterations or until convergence
    for iter in range(30) :
        # handle each wav files value separately
        for i in range (wav_values.shape[0]):
            min_dis = None # minimum distance from value to closet centroid
            centroid_row = 0 # save the centroid index whech relate to min_dis
            for j in range(centroids.shape[0]):
                # find the minimum distance between value point and centroid
                point_centro = np.array((centroids[j][0], centroids[j][1]))
                curr_dist = np.linalg.norm(wav_values[i] - point_centro)
                curr_dist = pow(curr_dist , 2)
                if min_dis is None or min_dis > curr_dist:
                    min_dis = curr_dist
                    centroid_row = j
            # update the values of the point to the related centroid
            centroids[centroid_row][2] += 1
            centroids[centroid_row][3] += wav_values[i][0]
            centroids[centroid_row][4] += wav_values[i][1]

        # update each centroids to be the average by the values related to
        flag_convergence = True
        # calculate the average value of each centroid according to his 3 columns that added
        for i in range(centroids.shape[0]):
            if centroids[i][2] != 0:
                x_val = centroids[i][3] / centroids[i][2]
                x_val_round = round(x_val,0)
                if x_val_round != centroids[i][0]:
                    flag_convergence = False
                    centroids[i][0] = x_val_round

                y_val = centroids[i][4] / centroids[i][2]
                y_val_round = round(y_val,0)
                if y_val_round != centroids[i][1]:
                    flag_convergence = False
                    centroids[i][1] = y_val_round

        new_centro = np.empty(shape=(centroids.shape[0], 2))
        for val in range(centroids.shape[0]):
            new_centro[val] = centroids[val, 0:2]
            centroids[val][2] = 0
            centroids[val][3] = 0
            centroids[val][4] = 0
        # write to output file the new values of the centroids
        output.write(f"[iter {iter}]:{','.join([str(i) for i in new_centro])}\n")
        if flag_convergence:
            break
    # update the file values to be the value of the values of closet centroid
    for i in range (wav_values.shape[0]):
        min_dis = None
        centroid_row = 0
        for j in range(centroids.shape[0]):
            point_centro = np.array((centroids[j][0], centroids[j][1]))
            curr_dist = np.linalg.norm(wav_values[i] - point_centro)
            curr_dist = pow(curr_dist , 2)
            if min_dis is None or min_dis > curr_dist:
                min_dis = curr_dist
                centroid_row = j
        new_values[i][0] = round(centroids[centroid_row][0],0)
        new_values[i][1] = round(centroids[centroid_row][1],0)
    scipy.io.wavfile.write("compressed.wav", samplerate, np.array(new_values, dtype=np.int16))


def main():
    sample, centroids = sys.argv[1], sys.argv[2]
    output = open("output.txt", "w+")
    compress_file(sample, centroids, output)


if __name__ == "__main__":
    main()
    

""" 
#function for report file:
sample, centroids = sys.argv[1], sys.argv[2]
output = None
if "2" in centroids:
    output = open("output2.txt", "w+")
else:
    output = open("output.txt", "w+")
loss = 0
k_amount=16
sum_min_dist = 0
samplerate, values = scipy.io.wavfile.read(sample)
wav_values = np.array(values.copy())
centroids = np.empty(shape=(k_amount, 2))
for k in range(k_amount):
    centroids[k] = np.random.randn(2)
ones_col = np.zeros((k_amount,3))
centroids = np.hstack((centroids, ones_col))
for iter in range(10):
    for i in range(wav_values.shape[0]):
        min_dis = None
        centroid_row = 0
        for j in range(k_amount):
            point_centro = np.array((centroids[j][0], centroids[j][1]))
            curr_dist = np.linalg.norm(wav_values[i] - point_centro)
            curr_dist = pow(curr_dist, 2)
            if min_dis is None or min_dis > curr_dist:
                min_dis = curr_dist
                centroid_row = j
        sum_min_dist += min_dis
        centroids[centroid_row][2] += 1
        centroids[centroid_row][3] += wav_values[i][0]
        centroids[centroid_row][4] += wav_values[i][1]
    print(iter)
    print(round(sum_min_dist/wav_values.shape[0]))
    sum_min_dist = 0
    # update each centroids to be the average by the values related to
    flag_convergence = True
    for i in range(centroids.shape[0]):
        if centroids[i][2] != 0:
            x_val = centroids[i][3] / centroids[i][2]
            x_val_round = round(x_val)
            if x_val_round != centroids[i][0]:
                flag_convergence = False
                centroids[i][0] = x_val_round

            y_val = centroids[i][4] / centroids[i][2]
            y_val_round = round(y_val)
            if y_val_round != centroids[i][1]:
                flag_convergence = False
                centroids[i][1] = y_val_round
"""
