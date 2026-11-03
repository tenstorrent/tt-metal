#!/bin/bash
# tt-media-server, Python (uvicorn) backend, Gio's spec settings, port 8102. Prints pid + log.
source "$(dirname "${BASH_SOURCE[0]}")/env_emb.sh"
L=$E/logs/media_py_$(date -u +%Y%m%dT%H%M%S).log
export MODEL=embeddinggemma-2 DEVICE=p150 MODEL_RUNNER=embedding_gemma2 USE_DYNAMIC_BATCHER=false DEVICE_IDS="${DEVICE_IDS:-(31)}"
export VLLM__MAX_NUM_BATCHED_TOKENS=8192 VLLM__MAX_MODEL_LENGTH=8192 VLLM__MIN_CONTEXT_LENGTH=32 VLLM__MAX_NUM_SEQS=32
export MAX_BATCH_SIZE=32 DEFAULT_THROTTLE_LEVEL=0 EMBEDDING_WARMUP_TIMEOUT_MS=1800000 API_KEY=${API_KEY:-your-secret-key}
export CACHE_ROOT=$E/cache_root PYTHONPATH="$ROOT:$TTIS/tt-media-server:$EMB_SCRIPTS"
cd $TTIS/tt-media-server
nohup uvicorn main:app --lifespan on --loop asyncio --host 127.0.0.1 --port 8102 > $L 2>&1 < /dev/null &
echo "pid $! log $L url http://127.0.0.1:8102"
