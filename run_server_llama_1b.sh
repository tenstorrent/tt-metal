#!/bin/bash

# Setup script to run Llama-3.2-1B inference server with tt-metal

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
    python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
fi

echo ""
echo "==== Starting Llama-3.2-1B inference server ===="
echo "Server will be available at: http://localhost:8000"
echo ""
echo "To test the server, run in another terminal:"
echo "  curl http://localhost:8000/v1/completions -H \"Content-Type: application/json\" -d '{\"model\": \"meta-llama/Llama-3.2-1B\", \"prompt\": \"San Francisco is a\", \"max_tokens\": 32, \"temperature\": 1}'"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the server
# For Llama-3.2-1B on a single device (N150/N300)
# Adjust MESH_DEVICE based on your hardware
cd $vllm_dir
VLLM_RPC_TIMEOUT=100000 MESH_DEVICE=N150 python examples/server_example_tt.py \
    --model "meta-llama/Llama-3.2-1B" \
    --max_num_seqs 16

echo "==== Server stopped ===="
