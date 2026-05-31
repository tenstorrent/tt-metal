#!/bin/bash
set -e

WEIGHTS_DIR="${1:-$HOME/liquid_weights}"
MODEL_ID="LiquidAI/LFM2.5-VL-1.6B"

echo "Downloading $MODEL_ID weights to $WEIGHTS_DIR..."
mkdir -p "$WEIGHTS_DIR"
huggingface-cli download "$MODEL_ID" --local-dir "$WEIGHTS_DIR" --local-dir-use-symlinks False
echo "Weights downloaded to $WEIGHTS_DIR"
echo "Set LIQUID_WEIGHTS=$WEIGHTS_DIR for inference"
