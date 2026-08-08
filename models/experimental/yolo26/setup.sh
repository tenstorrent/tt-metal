#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Setup script for YOLO26 model
# Downloads weights from Ultralytics

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="${SCRIPT_DIR}/weights"

echo "Setting up YOLO26 model..."

# Check if ultralytics is installed
if ! python3 -c "import ultralytics" 2>/dev/null; then
    echo "Installing ultralytics..."
    pip install ultralytics
fi

# Create weights directory
mkdir -p "${WEIGHTS_DIR}"

# Download YOLO26 variants
echo "Downloading YOLO26 weights..."

python3 << 'EOF'
import os
from ultralytics import YOLO

weights_dir = os.environ.get('WEIGHTS_DIR', 'weights')
os.makedirs(weights_dir, exist_ok=True)

# Download nano variant (smallest, good for initial bringup)
variants = ['yolo26n']  # Start with nano, add more later: 'yolo26s', 'yolo26m'

for variant in variants:
    print(f"Downloading {variant}...")
    try:
        model = YOLO(f"{variant}.pt")
        # Save the model state dict for our use
        import torch
        torch.save(model.model.state_dict(), f"{weights_dir}/{variant}.pth")
        print(f"Saved {variant} weights to {weights_dir}/{variant}.pth")
    except Exception as e:
        print(f"Warning: Could not download {variant}: {e}")
        print("You may need to download manually from Ultralytics")

print("Setup complete!")
EOF

echo "YOLO26 setup complete!"
echo "Weights saved to: ${WEIGHTS_DIR}/"
