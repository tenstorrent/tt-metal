#!/bin/bash

# Fixed script to run vLLM inference with proper environment setup
# This ensures all environment variables are set correctly

set -e

echo "==== vLLM TensTorrent Inference Setup ===="
echo ""

# Setup directories
export TT_METAL_HOME=/workspace/tt-metal-apv
export vllm_dir=/workspace/tt-vllm

# Source tt-metal environment
echo "📦 Setting up tt-metal environment..."
cd $TT_METAL_HOME
if [ -f env_vars_setup.sh ]; then
    source env_vars_setup.sh
    echo "✅ tt-metal environment loaded"
else
    echo "❌ Error: env_vars_setup.sh not found in $TT_METAL_HOME"
    exit 1
fi

# Source vLLM environment
echo ""
echo "📦 Setting up vLLM environment..."
cd $vllm_dir
if [ -f tt_metal/setup-metal.sh ]; then
    source tt_metal/setup-metal.sh
    echo "✅ vLLM environment loaded"
else
    echo "❌ Error: tt_metal/setup-metal.sh not found"
    exit 1
fi

# Verify critical environment variables
echo ""
echo "🔍 Environment check:"
echo "  VLLM_TARGET_DEVICE = $VLLM_TARGET_DEVICE"
echo "  TT_METAL_HOME = $TT_METAL_HOME"
echo "  PYTHONPATH = $PYTHONPATH"
echo "  PYTHON_ENV_DIR = $PYTHON_ENV_DIR"

if [ -z "$VLLM_TARGET_DEVICE" ]; then
    echo ""
    echo "❌ ERROR: VLLM_TARGET_DEVICE is not set!"
    echo "   This should be set to 'tt' by setup-metal.sh"
    exit 1
fi

if [ "$VLLM_TARGET_DEVICE" != "tt" ]; then
    echo ""
    echo "⚠️  WARNING: VLLM_TARGET_DEVICE is set to '$VLLM_TARGET_DEVICE'"
    echo "   Expected: 'tt' for TensTorrent devices"
fi

echo ""
echo "🚀 Running offline inference..."
echo ""

# Parse command line arguments or use defaults
MODEL="${1:-meta-llama/Llama-3.2-1B}"
DEVICE="${MESH_DEVICE:-N150}"

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Device: $DEVICE"
echo ""

# Run the inference
MESH_DEVICE=$DEVICE python examples/offline_inference_tt.py \
    --model "$MODEL" \
    "$@"

echo ""
echo "✅ Inference completed!"
