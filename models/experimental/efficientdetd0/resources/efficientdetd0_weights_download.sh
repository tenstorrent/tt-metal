#!/bin/bash

# Download your pretrained model:
# First try to download to resources folder, fallback to default location
RESOURCES_DIR="models/experimental/efficientdetd0/resources"
DEFAULT_DIR="models/experimental/efficientdetd0"

# Create directories if they don't exist
mkdir -p "$RESOURCES_DIR"
mkdir -p "$DEFAULT_DIR"

# Download to resources folder first
if wget https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d0.pth -O "$RESOURCES_DIR/efficientdet-d0.pth"; then
    echo "Downloaded weights to resources folder: $RESOURCES_DIR/efficientdet-d0.pth"
else
    # Fallback to default location
    wget https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d0.pth -O "$DEFAULT_DIR/efficientdet-d0.pth"
    echo "Downloaded weights to default location: $DEFAULT_DIR/efficientdet-d0.pth"
fi
