#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Script to download MonoDiffusion model weights

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WEIGHTS_DIR="${SCRIPT_DIR}/weights"

echo "================================================"
echo "MonoDiffusion Weights Download"
echo "================================================"

# Create weights directory
mkdir -p "${WEIGHTS_DIR}"

echo ""
echo "Downloading MonoDiffusion weights..."
echo "Note: This is a placeholder script."
echo "Actual weights should be downloaded from:"
echo "  - Official MonoDiffusion repository: https://github.com/ShuweiShao/MonoDiffusion"
echo "  - Or trained from scratch using the official training code"
echo ""

# Placeholder for actual download
# TODO: Add actual download commands when weights are available
# Example:
# wget -O "${WEIGHTS_DIR}/monodiffusion_kitti.pth" "https://example.com/weights/monodiffusion_kitti.pth"

echo "Weight download locations:"
echo "  KITTI weights:     ${WEIGHTS_DIR}/monodiffusion_kitti.pth"
echo "  Make3D weights:    ${WEIGHTS_DIR}/monodiffusion_make3d.pth"
echo ""

echo "================================================"
echo "Instructions:"
echo "================================================"
echo "1. Download pre-trained weights from MonoDiffusion repository"
echo "2. Place weights in: ${WEIGHTS_DIR}/"
echo "3. Supported weight files:"
echo "   - monodiffusion_kitti.pth (for KITTI dataset)"
echo "   - monodiffusion_make3d.pth (for Make3D dataset)"
echo ""
echo "Alternatively, train the model from scratch using:"
echo "  https://github.com/ShuweiShao/MonoDiffusion"
echo ""

# Create a dummy weight file for testing
if [ ! -f "${WEIGHTS_DIR}/monodiffusion_kitti.pth" ]; then
    echo "Creating dummy weight file for testing..."
    python3 << EOF
import torch
import os

weights_dir = "${WEIGHTS_DIR}"
os.makedirs(weights_dir, exist_ok=True)

# Create dummy state dict
dummy_state = {
    'encoder.conv1.weight': torch.randn(64, 3, 7, 7),
    'encoder.conv1.bias': torch.randn(64),
    'time_embed.0.weight': torch.randn(1024, 256),
    'time_embed.0.bias': torch.randn(1024),
    'time_embed.2.weight': torch.randn(256, 1024),
    'time_embed.2.bias': torch.randn(256),
}

torch.save(dummy_state, os.path.join(weights_dir, 'monodiffusion_kitti.pth'))
print(f"Created dummy weights at: {os.path.join(weights_dir, 'monodiffusion_kitti.pth')}")
EOF
fi

echo ""
echo "✓ Setup complete!"
echo ""
