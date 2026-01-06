#!/bin/bash
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

set -e

echo "Installing TT-SMI Python UI..."
echo

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install dependencies
echo
echo "Installing dependencies..."
pip install rich click pybind11

# Option to build with or without C++ bindings
if [ "$1" == "--pure-python" ]; then
    echo
    echo "Installing in pure Python mode (no C++ compilation)..."
    export TT_SMI_BUILD_NATIVE=0
else
    echo
    echo "Installing with C++ bindings..."
    export TT_SMI_BUILD_NATIVE=1
fi

# Install package
pip install -e .

echo
echo "Installation complete!"
echo
echo "Usage:"
echo "  tt-smi-ui             # Single snapshot"
echo "  tt-smi-ui -w          # Watch mode"
echo "  tt-smi-ui --help      # Show all options"
echo
echo "Python API:"
echo "  python3 -c 'from tt_smi import get_devices; print(get_devices())'"
echo
echo "Run tests:"
echo "  python3 test_python_ui.py"
