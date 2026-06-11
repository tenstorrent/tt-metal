#!/usr/bin/env bash
set -euo pipefail
cd /data/bzhang/tt-metal
source /data/bzhang/tt-metal/python_env/bin/activate

# Must MATCH the runner so the byte layout + chunk count line up.
export DEEPSEEK_PREFILL_TRACE_DIR=/mnt/models/deepseek-prefill-cache/golden/longbook_qa_eng_prefill_56320_nopad
export PREFILL_REQUEST_LOOP_PCC=1
export PREFILL_STANDALONE_CHUNKED_NCHUNKS=11
export PREFILL_STANDALONE_CHUNKED_SLOT=0
export PREFILL_H2D_SERVICE_ID=ds_prefill

export PREFILL_NUM_USERS=2
export PREFILL_SP=8
export PREFILL_TP=4
export PREFILL_MAX_SEQ_LEN=56320
export PREFILL_STANDALONE_ITERS=1
# Backup: wait up to 20 min for the descriptor (runner compile is ~8 min).
export PREFILL_H2D_CONNECT_TIMEOUT=1200

echo "=== producer env ==="
env | grep -E "^PREFILL_|^DEEPSEEK_" | sort
echo "=== launching prefill_h2d_producer.py ==="
exec python3 -m models.demos.deepseek_v3_d_p.tt.runners.prefill_h2d_producer
