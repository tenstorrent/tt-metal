#!/usr/bin/env bash
# Producer for the standalone request-mode runner (run_request_runner.sh):
# connects to the runner's H2D service and streams NCHUNKS chunks. Env must match the runner.
set -euo pipefail
METAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$METAL"
SITE="$(echo "$METAL"/python_env/lib/python*/site-packages 2>/dev/null | awk '{print $1}')"
export PYTHONPATH="${SITE:-}:$METAL:$METAL/ttnn:$METAL/tools${PYTHONPATH:+:$PYTHONPATH}"
PY="$METAL/python_env/bin/python3"; "$PY" -c '' 2>/dev/null || PY="$(command -v python3)"

export DEEPSEEK_PREFILL_TRACE_DIR="${DEEPSEEK_PREFILL_TRACE_DIR:-/mnt/models/deepseek-prefill-cache/golden/longbook_qa_eng_prefill_56320_nopad}"
export PREFILL_REQUEST_LOOP_PCC="${PREFILL_REQUEST_LOOP_PCC:-1}"
export PREFILL_STANDALONE_CHUNKED_NCHUNKS="${PREFILL_STANDALONE_CHUNKED_NCHUNKS:-11}"
export PREFILL_STANDALONE_CHUNKED_SLOT="${PREFILL_STANDALONE_CHUNKED_SLOT:-0}"
export PREFILL_H2D_SERVICE_ID="${PREFILL_H2D_SERVICE_ID:-ds_prefill}"
export PREFILL_NUM_USERS="${PREFILL_NUM_USERS:-2}"
export PREFILL_SP="${PREFILL_SP:-8}"
export PREFILL_TP="${PREFILL_TP:-4}"
export PREFILL_MAX_SEQ_LEN="${PREFILL_MAX_SEQ_LEN:-56320}"
export PREFILL_STANDALONE_ITERS="${PREFILL_STANDALONE_ITERS:-1}"
export PREFILL_H2D_CONNECT_TIMEOUT="${PREFILL_H2D_CONNECT_TIMEOUT:-1200}"

exec "$PY" -m models.demos.deepseek_v3_d_p.tt.runners.prefill_h2d_producer
