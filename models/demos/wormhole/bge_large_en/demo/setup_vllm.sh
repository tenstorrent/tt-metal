#!/bin/bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Script to set up BGE-Large-EN-v1.5 in vLLM repository
# Usage: ./setup_vllm.sh [path_to_vllm_repo]

set -e

VLLM_REPO="${1:-/home/ttuser/ashai/main/vllm}"
TT_FILE="$VLLM_REPO/vllm/platforms/tt.py"

if [ ! -f "$TT_FILE" ]; then
    echo "Error: vLLM file not found at $TT_FILE"
    echo "Please provide the correct path to vLLM repository:"
    echo "  ./setup_vllm.sh /path/to/vllm"
    exit 1
fi

echo "Setting up BGE-Large-EN-v1.5 in vLLM repository..."
echo "vLLM path: $VLLM_REPO"
echo ""

# Check if model is already in supported list
if grep -q "BAAI/bge-large-en-v1.5" "$TT_FILE"; then
    echo "✓ Model already in supported models list"
else
    echo "Adding model to supported models list..."
    # Find the line with the last model before the closing bracket
    # This is a simple approach - may need manual adjustment
    sed -i '/"deepseek-ai\/DeepSeek-R1-0528",/a\        "BAAI/bge-large-en-v1.5",' "$TT_FILE"
    echo "✓ Added BAAI/bge-large-en-v1.5 to supported models list"
fi

# Check if model is already registered
if grep -q "BGEForEmbedding" "$TT_FILE"; then
    echo "✓ Model already registered"
else
    echo "Registering model..."
    # Find the register_tt_models function and add registration before the end
    # Look for the last ModelRegistry.register_model call and add after it
    if grep -q "register_tt_models" "$TT_FILE"; then
    # Add registration after the last ModelRegistry.register_model call
    # Note: vLLM expects "TTBertModel" (constructed from architecture "BertModel")
    sed -i '/ModelRegistry.register_model.*gpt-oss/a\    # BGE Embedding Model\n    ModelRegistry.register_model(\n        "TTBertModel",\n        "models.demos.wormhole.bge_large_en.demo.generator_vllm:BGEForEmbedding",\n    )' "$TT_FILE"
        echo "✓ Registered BGEForEmbedding model"
    else
        echo "Warning: Could not find register_tt_models() function"
        echo "Please manually add the registration in register_tt_models() function:"
        echo ""
        echo "    ModelRegistry.register_model("
        echo "        \"BGEForEmbedding\","
        echo "        \"models.demos.wormhole.bge_large_en.demo.generator_vllm:BGEForEmbedding\","
        echo "    )"
    fi
fi

echo ""
echo "Setup complete! Please verify the changes:"
echo "  cd $VLLM_REPO"
echo "  git diff vllm/platforms/tt.py"
echo ""
echo "Then test the server:"
echo "  python -m vllm.entrypoints.openai.api_server --model BAAI/bge-large-en-v1.5 --tensor-parallel-size 1 --max-model-len 384"
