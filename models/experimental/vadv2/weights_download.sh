#!/bin/bash

# Check and install gdown
if ! command -v gdown &> /dev/null; then
    echo "gdown is not installed. Installing..."
    pip install gdown
else
    echo "gdown is already installed."
fi

# === Download and unzip ===

# Google Drive file ID for weights.zip
FILE_ID=""1yH8-4sNBsQJmon3fLdxugI0wJlaOB33s""
OUTPUT_FILE="models/experimental/vadv2/weights.zip"

# Ensure the target directory exists
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Download using gdown
echo "⬇️ Downloading weights.zip using gdown..."
gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "$OUTPUT_FILE"

# Check download success
if [ $? -eq 0 ]; then
    echo "✅ Download complete: $OUTPUT_FILE"
else
    echo "❌ Download failed!"
    exit 1
fi

# Show file type
echo "Verifying the downloaded file..."
file "$OUTPUT_FILE"

# Unzipping with Python
echo "📦 Unzipping the downloaded file using Python..."
python -c "
import zipfile
import os

zip_file = '$OUTPUT_FILE'
output_dir = 'models/experimental/vadv2/'

if zipfile.is_zipfile(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print(f'✅ Unzip complete: Files extracted to {output_dir}')
else:
    print('❌ Not a valid zip file!')
    exit(1)
"

# Check if unzip succeeded
if [ $? -eq 0 ]; then
    echo "✅ Unzip complete: Files extracted to models/experimental/vadv2/"
else
    echo "❌ Unzip failed!"
    exit 1
fi

# Clean up MacOS metadata
echo "🧹 Cleaning up __MACOSX and ._* files..."
find "models/experimental/vadv2/" -type d -name "__MACOSX" -exec rm -rf {} +
find "models/experimental/vadv2/" -type f -name "._*" -exec rm -f {} +

echo "✅ Cleanup complete."

# Delete the zip file
echo "🗑️ Deleting the zip file..."
rm -f "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo "✅ Zip file deleted: $OUTPUT_FILE"
else
    echo "⚠️ Failed to delete the zip file!"
fi
