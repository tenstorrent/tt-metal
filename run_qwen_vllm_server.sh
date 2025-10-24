#!/bin/bash

# Script to start vLLM server with Qwen3-32B model
# This server will be accessible at http://localhost:8000
# Based on the tt-metal README instructions

set -e

echo "==== Starting vLLM Server with Qwen3-32B ===="
echo ""

# Set default values
MODEL="${MODEL:-Qwen/Qwen3-32B}"
HF_MODEL="${MODEL:-Qwen/Qwen3-32B}"
# MESH_DEVICE="${MESH_DEVICE:-N150}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

echo "Configuration:"
echo "  Model: $MODEL"
echo "  HF Model: $HF_MODEL"
echo "  Device: $MESH_DEVICE"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo ""

echo "Starting vLLM server..."
echo "  - Server will be available at: http://localhost:$PORT"
echo "  - API endpoint: http://localhost:$PORT/v1"
echo "  - Health check: http://localhost:$PORT/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
# Using server_example_tt.py from the tt_metal examples
VLLM_RPC_TIMEOUT=100000 HF_MODEL=${HF_MODEL} \
    python examples/server_example_tt.py \
    --model "$MODEL" \
    --max_num_seqs 32 \
    --host "$HOST" \
    --port "$PORT"
