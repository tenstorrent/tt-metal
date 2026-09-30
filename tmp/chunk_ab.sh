#!/usr/bin/env bash
# One traced one-block run of the bit-exact set (8 KB payload + RoPE flags, now defaults) with the ring
# self-attn chunk pinned. Usage: chunk_ab.sh <stage_1|stage_2> <q,k|base>. Dump: tmp/blk/chunk_<stage>_<q>_<k>.pt
set -o pipefail
s=$1; c=$2
mkdir -p tmp/blk
source $PYTHON_ENV_DIR/bin/activate
[ "$c" = base ] || export LTX_SDPA_RING_CHUNK=$c
echo "=== CHUNK $s ${c}"
LTX_BLOCK_DUMP=tmp/blk/chunk_${s}_${c/,/_}.pt pytest -sv --timeout=1500 \
  models/tt_dit/tests/models/ltx/test_transformer_ltx.py -k "test_ltx_transformer_block_trace_perf and $s and 8k"
