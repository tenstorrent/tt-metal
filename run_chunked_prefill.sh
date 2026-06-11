#!/usr/bin/env bash
set -euo pipefail
cd /data/bzhang/tt-metal
source /data/bzhang/tt-metal/python_env/bin/activate

export MESH_DEVICE=TG
export DEEPSEEK_V3_HF_MODEL=/mnt/models/deepseek-ai/DeepSeek-R1-0528
export TT_DS_PREFILL_TTNN_CACHE=/mnt/models/DeepSeek-R1-0528-Cache/DeepSeek-R1-0528-Cache-prefill_secure
export TT_DS_PREFILL_HOST_REF_CACHE=/mnt/models/deepseek-prefill-cache/golden
export DEEPSEEK_PREFILL_TRACE_DIR=/mnt/models/deepseek-prefill-cache/golden/longbook_qa_eng_prefill_56320_nopad

# chunked standalone mode: golden longbook_qa input, per-layer KV-cache PCC vs trace
export PREFILL_STANDALONE_CHUNKED=1
export PREFILL_NUM_USERS=${PREFILL_NUM_USERS:-8}     # was 2; bumped to 8 users
export PREFILL_STANDALONE_CHUNKED_ALL_SLOTS=1        # prefill + PCC-validate ALL 8 user slots (not just slot 0)
export PREFILL_STANDALONE_CHUNKED_NCHUNKS=11         # 11 chunks
export PREFILL_CHUNK_SIZE=5120                       # 5*1024 tokens/chunk -> 11*5120 = 56320 (~55k) per user
export PREFILL_SP=8
export PREFILL_TP=4
export PREFILL_MAX_SEQ_LEN=56320

echo "=== env ==="
env | grep -E "MESH_DEVICE|^PREFILL_|^DEEPSEEK_|^TT_DS_" | sort
echo "=== launching prefill_runner.py ==="
exec python3 models/demos/deepseek_v3_d_p/tt/runners/prefill_runner.py
