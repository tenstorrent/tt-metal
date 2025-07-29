# This script automatically download the model weights from Google Drive

#!/bin/bash

# Output filename
OUTPUT="models/demos/vgg_unet/vgg_unet_torch.pth"

# Create output directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT")"

if ! command -v gdown &> /dev/null
then
    echo "Installing gdown..."
    pip install gdown
fi

FILE_ID="11Uj52GnFHtgPSgfiUIY8HOG3MGoDQFp0"
echo "Downloading file to $OUTPUT..."
gdown "https://drive.google.com/uc?id=${FILE_ID}" --output "$OUTPUT"
