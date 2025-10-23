#!/bin/bash

# Setup script to run Llama-3.2-1B offline inference with tt-metal

set -e

# Set up directories
export vllm_dir=/workspace/tt-vllm
export TT_METAL_HOME=/workspace/tt-metal-apv

echo "==== Setting up tt-metal environment ===="
cd $TT_METAL_HOME
source $TT_METAL_HOME/env_vars_setup.sh

echo "==== Setting up vllm environment ===="
cd $vllm_dir
source $vllm_dir/tt_metal/setup-metal.sh

echo "Environment setup:"
echo "  TT_METAL_HOME=$TT_METAL_HOME"
echo "  VLLM_TARGET_DEVICE=$VLLM_TARGET_DEVICE"
echo "  PYTHON_ENV_DIR=$PYTHON_ENV_DIR"
echo "  PYTHONPATH=$PYTHONPATH"

# Check if vllm is installed
echo "==== Checking vllm installation ===="
if ! python -c "import vllm" 2>/dev/null; then
    echo "vLLM not found, installing..."
    pip3 install --upgrade pip
    cd $vllm_dir && pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu
else
    echo "vLLM is already installed"
fi

echo "==== Running Llama-3.2-1B offline inference ===="
# For Llama-3.2-1B on a single device (N150/N300)
# Adjust MESH_DEVICE based on your hardware
cd $vllm_dir
MESH_DEVICE=N150 python examples/offline_inference_tt.py --model "meta-llama/Llama-3.2-1B"

echo "==== Done! ===="
