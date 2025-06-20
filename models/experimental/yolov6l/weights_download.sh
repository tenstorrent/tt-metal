# This script automatically download the model weights from Google Drive

#!/bin/bash

# Output filename
OUTPUT="tests/ttnn/integration_tests/yolov6l/yolov6l.pt"

# Create output directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT")"

# Download the file using wget
if wget "https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6l.pt" -O "${OUTPUT}"; then
    echo "File downloaded successfully: ${OUTPUT}"
else
    echo "Error downloading the file."
    exit 1
fi
