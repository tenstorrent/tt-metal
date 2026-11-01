#!/bin/bash
# Iteration 26c sweep: chain length x hop loads (position-aware LPT) at the TP=4 shard shape.
set -u
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
ROOT=/home/ttuser/agobeljic/tt-metal
cd $ROOT/tt-train
export TT_METAL_HOME=$ROOT TT_METAL_RUNTIME_ROOT=$ROOT
unset TT_MESH_GRAPH_DESC_PATH
run() {  # tag chain fw q kv
  rm -f /dev/shm/sm_segment.* 2>/dev/null
  TTML_SDPA_CHAIN_LENGTH=$2 TTML_SDPA_FW_HOP_LOAD=$3 TTML_SDPA_Q_HOP_LOAD=$4 TTML_SDPA_KV_HOP_LOAD=$5 timeout 300 ../python_env/bin/python $O/iter26_sdpa_time.py --tag "$1" --iters 20 > $O/iter26c_sweep_$1.log 2>&1; rc=$?
  grep -E "^$1 case" $O/iter26c_sweep_$1.log || echo "$1: rc=$rc (no result)"
}
echo "=== sweep start $(date -u +%FT%TZ)"
run L9_h0 9 0 0 0
run L11_h0 11 0 0 0
run L12_h0 12 0 0 0
run L8b_h0 8 0 0 0
run L14b_h0 14 0 0 0
echo "=== sweep done $(date -u +%FT%TZ)"
