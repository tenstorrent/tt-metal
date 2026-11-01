#!/bin/bash
# Iteration 27: chain-length sweep with the per-round stream bounds (forward / backward knobs), TP=4 shard shape.
set -u
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
ROOT=/home/ttuser/agobeljic/tt-metal
cd $ROOT/tt-train
export TT_METAL_HOME=$ROOT TT_METAL_RUNTIME_ROOT=$ROOT
unset TT_MESH_GRAPH_DESC_PATH
run() {  # tag fw_len bw_len
  rm -f /dev/shm/sm_segment.* 2>/dev/null
  TTML_SDPA_FW_CHAIN_LENGTH=$2 TTML_SDPA_BW_CHAIN_LENGTH=$3 timeout 300 ../python_env/bin/python $O/iter26_sdpa_time.py --tag "$1" --iters 20 > $O/iter27_sweep_$1.log 2>&1; rc=$?
  grep -E "^$1 case" $O/iter27_sweep_$1.log || echo "$1: rc=$rc (no result)"
}
for fw in 6 8 10; do for bw in 7 10 14; do run fw${fw}_bw${bw} $fw $bw; done; done
run fw8_bw14b 8 14
