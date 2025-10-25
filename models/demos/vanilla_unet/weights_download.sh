# This script automatically download the model weights from Google Drive

#!/bin/bash

# Output filename
OUTPUT="models/demos/vanilla_unet/unet.pt"

# Create output directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT")"

# Download the file using wget
if wget "https://github.com/mateuszbuda/brain-segmentation-pytorch/raw/master/weights/unet.pt" -O "${OUTPUT}"; then
    echo "File downloaded successfully: ${OUTPUT}"
else
    echo "Error downloading the file."
    exit 1
fi
