import numpy as np
import matplotlib.pyplot as plt
import sys


# Get the command-line argument: csv file name
csv_file_name = sys.argv[1]
csv_file_name += ".csv"
print("\ncvs file name:")
print(csv_file_name)

png_file_name = sys.argv[2]
png_file_name += ".png"
print("\npng file name:")
print(png_file_name)


'''
data = np.loadtxt(csv_file_name, delimiter=',')
print(data)
print(data.shape)
heatmap = plt.imshow(data, cmap='Spectral')
plt.colorbar()
plt.show()
plt.savefig(png_file_name)
'''

# Load the data from the CSV file
data = np.loadtxt(csv_file_name, delimiter=',')

# Create a figure with two subplots
fig, axs = plt.subplots(2,2, figsize=(10, 5))

# Plot the data on each subplot
heatmap1 = axs[0,0].imshow(data, cmap='Spectral')
axs[0,0].set_title('Heatmap 1')
fig.colorbar(heatmap1, ax=axs[0])

heatmap2 = axs[0,1].imshow(data, cmap='Spectral')
axs[1].set_title('Heatmap 2')
fig.colorbar(heatmap2, ax=axs[1])

heatmap3 = axs[2].imshow(data, cmap='Spectral')
axs[2].set_title('Heatmap 2')
fig.colorbar(heatmap3, ax=axs[2])

# Save the figure as a PNG file
plt.savefig(png_file_name)




'''
data = [1, 2, 3, 4, 5, 6, 7, 8, 10]

# Convert the list to a 2D numpy array
data_array = np.reshape(data, (3, 3))
heatmap = plt.imshow(data_array, cmap='Spectral')
plt.colorbar()
plt.show()
plt.savefig('plot1.png')

b = data_array.transpose()
heatmap = plt.imshow(b, cmap='Spectral')
plt.colorbar()
plt.savefig('plot2.png')
'''
