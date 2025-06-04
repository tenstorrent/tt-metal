#!/bin/bash

if ! python -c "import evaluate" &> /dev/null; then
    echo "'evaluate' library not found, installing..."
    pip install evaluate
else
    echo "'evaluate' library is already installed."
fi


if ! command -v wget &> /dev/null; then
    echo "wget is not installed. Installing..."
    sudo apt-get update
    sudo apt-get install -y wget
fi


# No need to check for 'unzip' anymore since we will use Python for unzipping


ONEDRIVE_ZIP_LINK="https://tinyurl.com/4xtuxr3k"


OUTPUT_FILE="models/demos/segformer/demo/validation_data.zip"

mkdir -p "$(dirname "$OUTPUT_FILE")"

# Download the zip file using wget
echo "Downloading file from OneDrive..."
wget --trust-server-names --content-disposition -O "$OUTPUT_FILE" "$ONEDRIVE_ZIP_LINK"

# Check if the download was successful
if [ $? -eq 0 ]; then
    echo "Download complete: $OUTPUT_FILE"
else
    echo "Download failed!"
    exit 1
fi


echo "Verifying the downloaded file..."
file "$OUTPUT_FILE"


# Unzipping the downloaded file using Python's zipfile module
echo "Unzipping the downloaded file using Python..."
python -c "
import zipfile
import os

zip_file = '$OUTPUT_FILE'
output_dir = 'models/demos/segformer/demo/'

if zipfile.is_zipfile(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print(f'Unzip complete: Files extracted to {output_dir}')
else:
    print('Not a valid zip file!')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo "Unzip complete: Files extracted to models/demos/segformer/demo/"
else
    echo "Unzip failed!"
    exit 1
fi

# Clean up any __MACOSX and ._* files (MacOS metadata)
echo "Cleaning up __MACOSX and ._* files..."
find "models/demos/segformer/demo/" -type d -name "__MACOSX" -exec rm -rf {} +
find "models/demos/segformer/demo/" -type f -name "._*" -exec rm -f {} +

echo "Cleanup complete."
