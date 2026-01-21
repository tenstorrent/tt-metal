#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Setup script for downloading OpenVLA weights from HuggingFace
#
# Usage:
#   ./setup_weights.sh              # Downloads to ~/openvla_weights/
#   ./setup_weights.sh /path/to/dir # Downloads to specified directory
#
# After downloading, set the environment variable:
#   export OPENVLA_WEIGHTS=~/openvla_weights/

set -e

SAVE_PATH="${1:-$HOME/openvla_weights}"

echo "=========================================="
echo "OpenVLA Weights Setup"
echo "=========================================="
echo "Download location: $SAVE_PATH"
echo ""

# Create directory
mkdir -p "$SAVE_PATH"

# Download using huggingface_hub
export SAVE_PATH
python3 -c '
from huggingface_hub import snapshot_download
import os

save_path = os.environ["SAVE_PATH"]
print(f"Downloading OpenVLA weights to: {save_path}")
print("This may take a while (~14GB)...")
print("")

path = snapshot_download(
    repo_id="openvla/openvla-7b",
    local_dir=save_path,
    allow_patterns=["*.safetensors", "*.json", "*.txt", "*.py"],
    ignore_patterns=["*.bin", "*.msgpack"],
)
print("")
print(f"Downloaded to: {path}")
'

echo ""
echo "=========================================="
echo "Setup complete!"
echo ""
echo "Add this to your environment:"
echo "  export OPENVLA_WEIGHTS=$SAVE_PATH"
echo "=========================================="
