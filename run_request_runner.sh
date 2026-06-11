#!/usr/bin/env bash
set -euo pipefail
cd /data/bzhang/tt-metal
source /data/bzhang/tt-metal/python_env/bin/activate

export MESH_DEVICE=TG
export DEEPSEEK_V3_HF_MODEL=/mnt/models/deepseek-ai/DeepSeek-R1-0528
export TT_DS_PREFILL_TTNN_CACHE=/mnt/models/DeepSeek-R1-0528-Cache/DeepSeek-R1-0528-Cache-prefill_secure
export TT_DS_PREFILL_HOST_REF_CACHE=/mnt/models/deepseek-prefill-cache/golden
export DEEPSEEK_PREFILL_TRACE_DIR=/mnt/models/deepseek-prefill-cache/golden/longbook_qa_eng_prefill_56320_nopad

# REQUEST MODE: neither PREFILL_STANDALONE_CHUNKED nor PREFILL_STANDALONE is set,
# so main() falls through to building the H2D socket service + run_request_loop.
# Bounded PCC mode: run for NCHUNKS pushes from the producer, then PCC-check the KV cache.
export PREFILL_REQUEST_LOOP_PCC=1
export PREFILL_STANDALONE_CHUNKED_NCHUNKS=11
export PREFILL_STANDALONE_CHUNKED_SLOT=0
export PREFILL_H2D_SERVICE_ID=ds_prefill

export PREFILL_NUM_USERS=2
export PREFILL_SP=8
export PREFILL_TP=4
export PREFILL_MAX_SEQ_LEN=56320

echo "=== runner env ==="
env | grep -E "MESH_DEVICE|^PREFILL_|^DEEPSEEK_|^TT_DS_" | sort
echo "=== launching prefill_runner.py (REQUEST MODE) ==="
exec python3 models/demos/deepseek_v3_d_p/tt/runners/prefill_runner.py
