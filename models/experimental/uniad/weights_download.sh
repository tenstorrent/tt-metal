#!/bin/bash

OUTPUT_FILE="models/experimental/uniad/uniad_base_e2e.pth"

# Ensure the target directory exists
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Download the file
if wget "https://github.com/OpenDriveLab/UniAD/releases/download/v1.0.1/uniad_base_e2e.pth" -O "$OUTPUT_FILE"; then
    echo "✅ File downloaded successfully: $OUTPUT_FILE"
else
    echo "❌ Download failed!"
    exit 1
fi

# Show file type
echo "Verifying the downloaded file..."
file "$OUTPUT_FILE"
