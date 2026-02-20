#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Setup script for YUNet model
# Clones the original YUNet PyTorch repository and sets up weights

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YUNET_DIR="$SCRIPT_DIR/YUNet"

echo "Setting up YUNet model..."

# Clone YUNet repo if not exists
if [ ! -d "$YUNET_DIR" ]; then
    echo "Cloning YUNet repository..."
    git clone https://github.com/jahongir7174/YUNet.git "$YUNET_DIR"
else
    echo "YUNet directory already exists, skipping clone."
fi

# Create __init__.py files to make it a proper Python package
echo "Setting up Python package structure..."
touch "$YUNET_DIR/__init__.py"
touch "$YUNET_DIR/nets/__init__.py"

# Check if weights exist
if [ -f "$YUNET_DIR/weights/best.pt" ]; then
    echo "Weights already present at $YUNET_DIR/weights/best.pt"
else
    echo "WARNING: Weights not found at $YUNET_DIR/weights/best.pt"
    echo "Please download weights from: https://github.com/jahongir7174/YUNet/releases"
    echo "Or train the model following instructions in the YUNet repository."
fi

echo ""
echo "Setup complete!"
echo "YUNet directory: $YUNET_DIR"
echo ""
echo "To run tests:"
echo "  pytest models/experimental/yunet/tests/pcc/test_pcc.py -v"
echo ""
