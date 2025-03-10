#!/bin/bash

# Check if wget is installed
if ! command -v wget &> /dev/null; then
    echo "wget is not installed. Installing..."
    sudo apt-get update
    sudo apt-get install -y wget
fi

echo "Downloading file from OneDrive..."

# Define the direct OneDrive download link
ONEDRIVE_LINK="https://multicorewareinc1-my.sharepoint.com/:u:/g/personal/venkatesh_guduru_multicorewareinc_com/EUy7ascSaDlKq1CJ-WxMbF8BpuHouuBToVlu8lFR3MEMnQ?download=1"
DATA_FOLDER="models/demos/segformer/demo/validation.zip"

# Download the zip file from OneDrive using wget
wget "$ONEDRIVE_LINK" -O "$DATA_FOLDER"

# Check if unzip is installed
if ! command -v unzip &> /dev/null; then
    echo "unzip command is not available. Installing unzip..."
    sudo apt-get update
    sudo apt-get install -y unzip
fi

# Define the output directory
OUTPUT_DIRECTORY="models/demos/segformer/demo/validation_data"
mkdir -p "$OUTPUT_DIRECTORY"

# Unzip the downloaded file into the output directory
unzip "$DATA_FOLDER" -d "$OUTPUT_DIRECTORY"
