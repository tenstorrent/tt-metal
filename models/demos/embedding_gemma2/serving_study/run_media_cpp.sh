#!/bin/bash
# tt-media-server, C++ (cpp_server) backend as run.py's docker path configures it, port 8103. Prints pid + log.
# DEVICE_IDS pins the chip: both backends set each worker's TT_VISIBLE_DEVICES from it (31 = $EMB_DEV = 86:00.0).
source "$(dirname "${BASH_SOURCE[0]}")/env_emb.sh"
L=$E/logs/media_cpp_$(date -u +%Y%m%dT%H%M%S).log
export SERVER_MODE=cpp MODEL_SERVICE=embedding MODEL_RUNNER_TYPE=tt_embedding_gemma2 DEVICE=p150 DEVICE_IDS="${DEVICE_IDS:-(31)}"
export VLLM__MAX_NUM_BATCHED_TOKENS=8192 VLLM__MAX_MODEL_LENGTH=8192 VLLM__MIN_CONTEXT_LENGTH=32 VLLM__MAX_NUM_SEQS=32
export MAX_BATCH_SIZE=32 DEFAULT_THROTTLE_LEVEL=0 USE_DYNAMIC_BATCHER=false EMBEDDING_WARMUP_TIMEOUT_MS=1800000
export OPENAI_API_KEY=${API_KEY:-your-secret-key} CACHE_ROOT=$E/cache_root
export PYTHONPATH="$ROOT:$TTIS/tt-media-server:$EMB_SCRIPTS"
# The binary links /usr libpython3.10 (apt), but Gio's venv is uv's standalone CPython, whose stdlib expects _ctypes
# etc. built into its own libpython: load uv's libpython instead (same soname, LD_LIBRARY_PATH beats RUNPATH).
export LD_LIBRARY_PATH=${UV_PYTHON_LIB:-/home/ttuser/.local/share/uv/python/cpython-3.10.19-linux-x86_64-gnu/lib}:$LD_LIBRARY_PATH
cd $TTIS/tt-media-server/cpp_server
nohup ./build/tt_media_server_cpp -p 8103 > $L 2>&1 < /dev/null &
echo "pid $! log $L url http://127.0.0.1:8103"
