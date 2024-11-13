#!/bin/bash

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "gdown is not installed. Installing..."
    pip install gdown
fi
echo "I'm in...."

DATA_PATH_ID="1OW_3sGR05J4rvOuttddWYWCIJzR_2ri-"
DATA_FOLDER="models/demos/segformer/demo/validation.zip"

gdown "https://drive.google.com/uc?id=${DATA_PATH_ID}" -O "${DATA_FOLDER}"

if ! command -v unzip &> /dev/null; then
    echo "unzip command is not available. Installing unzip..."
    sudo apt-get update
    sudo apt-get install -y unzip
fi

OUTPUT_DIRECTORY="models/demos/segformer/demo/validation_data"
mkdir -p "$OUTPUT_DIRECTORY"
unzip "$DATA_FOLDER" -d "$OUTPUT_DIRECTORY"
