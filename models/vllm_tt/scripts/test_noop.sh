#!/bin/bash
# Test the TT plugin with the no-op model.
# Run from tt-metal root: bash models/vllm_tt/scripts/test_noop.sh
set -euo pipefail

export VLLM_TARGET_DEVICE=tt
export VLLM_LOGGING_LEVEL=INFO

MODEL_DIR="models/vllm_test_utils/no_op_test_plugin"
TOKENIZER="meta-llama/Llama-3.1-8B-Instruct"
PORT=8192
SERVER_LOG="$(mktemp)"
SERVER_PID=""

cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    echo "Server log: $SERVER_LOG"
}
trap cleanup EXIT

echo "=== Starting vLLM server ==="
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --tokenizer "$TOKENIZER" \
    --load-format dummy \
    --max-model-len 4096 \
    --max-num-seqs 32 \
    --block-size 64 \
    --num-gpu-blocks-override 2048 \
    --port "$PORT" \
    --no-enable-log-requests \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "Server PID: $SERVER_PID, log: $SERVER_LOG"
echo "Waiting for server to be ready..."

for i in $(seq 1 60); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Server process died. Last 50 lines of log:"
        tail -50 "$SERVER_LOG"
        exit 1
    fi
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "Server failed to start within 60s. Last 50 lines of log:"
        tail -50 "$SERVER_LOG"
        exit 1
    fi
    sleep 1
done

echo ""
echo "=== Running benchmark ==="
vllm bench serve \
    --base-url "http://localhost:${PORT}" \
    --model "$MODEL_DIR" \
    --tokenizer "$TOKENIZER" \
    --dataset-name random \
    --random-input-len 50 \
    --random-output-len 50 \
    --num-prompts 16 \
    --ignore-eos

echo ""
echo "=== Test passed ==="
