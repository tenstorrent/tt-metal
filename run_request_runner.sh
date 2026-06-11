#!/usr/bin/env bash
# Standalone request-mode runner (H2D socket + bounded KV PCC).
# Pairs with run_request_producer.sh: producer streams NCHUNKS chunks, then KV is PCC-checked.
# Paths derive from this script; override via env.
set -euo pipefail
METAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$METAL"
SITE="$(echo "$METAL"/python_env/lib/python*/site-packages 2>/dev/null | awk '{print $1}')"
export PYTHONPATH="${SITE:-}:$METAL:$METAL/ttnn:$METAL/tools${PYTHONPATH:+:$PYTHONPATH}"
PY="$METAL/python_env/bin/python3"; "$PY" -c '' 2>/dev/null || PY="$(command -v python3)"

export MESH_DEVICE="${MESH_DEVICE:-TG}"
export DEEPSEEK_V3_HF_MODEL="${DEEPSEEK_V3_HF_MODEL:-/mnt/models/deepseek-ai/DeepSeek-R1-0528}"
export TT_DS_PREFILL_TTNN_CACHE="${TT_DS_PREFILL_TTNN_CACHE:-/mnt/models/DeepSeek-R1-0528-Cache/DeepSeek-R1-0528-Cache-prefill_secure}"
export TT_DS_PREFILL_HOST_REF_CACHE="${TT_DS_PREFILL_HOST_REF_CACHE:-/mnt/models/deepseek-prefill-cache/golden}"
export DEEPSEEK_PREFILL_TRACE_DIR="${DEEPSEEK_PREFILL_TRACE_DIR:-/mnt/models/deepseek-prefill-cache/golden/longbook_qa_eng_prefill_56320_nopad}"
# Request mode = no PREFILL_STANDALONE*; bounded PCC checks the KV cache after NCHUNKS.
export PREFILL_REQUEST_LOOP_PCC="${PREFILL_REQUEST_LOOP_PCC:-1}"
export PREFILL_STANDALONE_CHUNKED_NCHUNKS="${PREFILL_STANDALONE_CHUNKED_NCHUNKS:-11}"
export PREFILL_STANDALONE_CHUNKED_SLOT="${PREFILL_STANDALONE_CHUNKED_SLOT:-0}"
export PREFILL_H2D_SERVICE_ID="${PREFILL_H2D_SERVICE_ID:-ds_prefill}"
export PREFILL_NUM_USERS="${PREFILL_NUM_USERS:-2}"
export PREFILL_SP="${PREFILL_SP:-8}"
export PREFILL_TP="${PREFILL_TP:-4}"
export PREFILL_MAX_SEQ_LEN="${PREFILL_MAX_SEQ_LEN:-56320}"

exec "$PY" models/demos/deepseek_v3_d_p/tt/runners/prefill_runner.py
