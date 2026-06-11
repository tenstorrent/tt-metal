#!/usr/bin/env bash
# Standalone chunked prefill + per-slot golden KV-cache PCC (no engine, no H2D socket).
# Portable: repo path derives from this script's location; override paths via env.
# Defaults: 8 users, 11x5120 = 56320 (~55k) per user, all slots validated.
set -euo pipefail
METAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$METAL"
# Prefer the venv python; fall back to system python3 + venv site-packages on PYTHONPATH.
SITE="$(echo "$METAL"/python_env/lib/python*/site-packages 2>/dev/null | awk '{print $1}')"
export PYTHONPATH="${SITE:-}:$METAL:$METAL/ttnn:$METAL/tools${PYTHONPATH:+:$PYTHONPATH}"
PY="$METAL/python_env/bin/python3"; "$PY" -c '' 2>/dev/null || PY="$(command -v python3)"

export MESH_DEVICE="${MESH_DEVICE:-TG}"
export DEEPSEEK_V3_HF_MODEL="${DEEPSEEK_V3_HF_MODEL:-/mnt/models/deepseek-ai/DeepSeek-R1-0528}"
export TT_DS_PREFILL_TTNN_CACHE="${TT_DS_PREFILL_TTNN_CACHE:-/mnt/models/DeepSeek-R1-0528-Cache/DeepSeek-R1-0528-Cache-prefill_secure}"
export TT_DS_PREFILL_HOST_REF_CACHE="${TT_DS_PREFILL_HOST_REF_CACHE:-/mnt/models/deepseek-prefill-cache/golden}"
# Populated 56320 longbook_qa golden trace (metadata.json + kv_cache/).
export DEEPSEEK_PREFILL_TRACE_DIR="${DEEPSEEK_PREFILL_TRACE_DIR:-/mnt/models/deepseek-prefill-cache/golden/longbook_qa_eng_prefill_56320_nopad}"

export PREFILL_STANDALONE_CHUNKED=1
export PREFILL_NUM_USERS="${PREFILL_NUM_USERS:-8}"
export PREFILL_STANDALONE_CHUNKED_ALL_SLOTS="${PREFILL_STANDALONE_CHUNKED_ALL_SLOTS:-1}"
export PREFILL_STANDALONE_CHUNKED_NCHUNKS="${PREFILL_STANDALONE_CHUNKED_NCHUNKS:-11}"
export PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE:-5120}"
export PREFILL_SP="${PREFILL_SP:-8}"
export PREFILL_TP="${PREFILL_TP:-4}"
export PREFILL_MAX_SEQ_LEN="${PREFILL_MAX_SEQ_LEN:-56320}"

echo "[chunked] tt_metal=$METAL python=$PY users=$PREFILL_NUM_USERS"
env | grep -E "MESH_DEVICE|^PREFILL_|^DEEPSEEK_|^TT_DS_" | sort
exec "$PY" models/demos/deepseek_v3_d_p/tt/runners/prefill_runner.py
