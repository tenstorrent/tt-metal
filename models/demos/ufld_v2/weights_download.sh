#!/bin/bash

# Check if gdown is installed, install it if not
if ! python -c "import gdown" &> /dev/null; then
    echo "gdown not found, installing..."
    pip install gdown
else
    echo "gdown is already installed."
fi

# Output filename
OUTPUT="models/demos/ufld_v2/tusimple_res34.pth"

# Create output directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT")"

# Google Drive URL and File ID for downloading
google_drive_url="https://drive.google.com/file/d/1pkz8homK433z39uStGK3ZWkDXrnBAMmX/view"
file_id=$(echo $google_drive_url | grep -oP "(?<=/d/)[^/]+")

google_drive_download_url="https://drive.google.com/uc?id=$file_id"

# Download the file using gdown
if gdown "$google_drive_download_url" -O "${OUTPUT}"; then
    echo "File downloaded successfully: ${OUTPUT}"
else
    echo "Error downloading the file."
    exit 1
fi
