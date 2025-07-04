# This script automatically download the model weights

#!/bin/bash

# Output filename
OUTPUT="models/experimental/centernet/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth"

# Create output directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT")"

# Download the file using wget
if wget "https://download.openmmlab.com/mmdetection/v2.0/centernet/centernet_resnet18_dcnv2_140e_coco/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth" -O "${OUTPUT}"; then
    echo "File downloaded successfully: ${OUTPUT}"
else
    echo "Error downloading the file."
    exit 1
fi
