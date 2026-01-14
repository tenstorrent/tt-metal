#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# YUNet Setup Script
# This script clones the YUNet repository and makes it importable.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YUNET_DIR="${SCRIPT_DIR}/YUNet"

echo "=== YUNet Setup ==="
echo "Target directory: ${YUNET_DIR}"

# Clone YUNet repository
if [ -d "${YUNET_DIR}" ]; then
    echo "YUNet directory already exists. Skipping clone."
else
    echo "Cloning YUNet repository..."
    git clone https://github.com/jahongir7174/YUNet.git "${YUNET_DIR}"
    echo "Clone complete."
fi

# Add __init__.py files to make it importable as Python package
echo "Adding __init__.py files for Python imports..."
touch "${YUNET_DIR}/__init__.py"
touch "${YUNET_DIR}/nets/__init__.py"
touch "${YUNET_DIR}/utils/__init__.py" 2>/dev/null || true

# Create weights directory
WEIGHTS_DIR="${YUNET_DIR}/weights"
mkdir -p "${WEIGHTS_DIR}"

# Check for weights
WEIGHTS_FILE="${WEIGHTS_DIR}/best.pt"
if [ -f "${WEIGHTS_FILE}" ]; then
    echo "Weights file found."
else
    echo ""
    echo "*** WEIGHTS NOT FOUND ***"
    echo "Please download/train weights and place at:"
    echo "  ${WEIGHTS_FILE}"
    echo ""
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Run tests:"
echo "  pytest models/experimental/yunet/tests/pcc/test_pcc.py -v"
echo "  pytest models/experimental/yunet/tests/perf/test_yunet_perf.py -v"
echo ""
echo "Run demo:"
echo "  python -m models.experimental.yunet.demo.demo -i <image.jpg> -o <result.jpg>"
