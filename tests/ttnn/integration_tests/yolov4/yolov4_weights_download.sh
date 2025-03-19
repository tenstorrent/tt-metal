#!/bin/bash

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "gdown is not installed. Installing..."
    pip3 install gdown
fi

# Google Drive file ID
FILE_ID="1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ"
# Output filename
OUTPUT="tests/ttnn/integration_tests/yolov4/yolov4.pth"

# Download the file
python3 -m gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "${OUTPUT}"
