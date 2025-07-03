# This script automatically download the model weights from Google Drive

#!/bin/bash

# Output filename
OUTPUT="models/demos/mobilenetv2/mobilenet_v2-b0353104.pth"

# Create output directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT")"

# Download the file using wget
if wget "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth" -O "${OUTPUT}"; then
    echo "File downloaded successfully: ${OUTPUT}"
else
    echo "Error downloading the file."
    exit 1
fi
