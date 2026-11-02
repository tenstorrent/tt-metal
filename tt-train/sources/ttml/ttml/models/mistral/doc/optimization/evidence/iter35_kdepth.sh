#!/bin/bash
# Iteration 35: K-block depth (in0_block_w) of the gate_up forward at the TP shard shape on the 11x10 grid, which the
# iteration-34 sweep never measured above 3 (its L1 estimate counted the double buffering twice).
set -u
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
R=/home/ttuser/agobeljic/tt-metal
cd $R/tt-train
export TT_METAL_HOME=$R TT_METAL_RUNTIME_ROOT=$R
unset TT_MESH_GRAPH_DESC_PATH
PY=../python_env/bin/python
run() {  # tag args
  rm -f /dev/shm/sm_segment.* 2>/dev/null
  echo "=== $1 $(date -u +%FT%TZ)"
  timeout 2400 $PY $O/iter35_matmul_sweep1d.py "${@:2}" > $O/iter35_sweep_$1.log 2>&1; echo "rc=$?"
  grep -E "ttnn's own|configs tried" $O/iter35_sweep_$1.log; sed -n '/--- sorted ---/,$p' $O/iter35_sweep_$1.log | grep " ms " | head -8
}
run gateup_fwd --m 1024 --k 5184 --n 16384 --grids 11x10 --in0-block-w 2,3,6,9,18,27 --subs 4x1,2x2 --out-blocks 16x5,32x5,8x5 --reps 5
run gateup_fwd_noacc --m 1024 --k 5184 --n 16384 --grids 11x10 --in0-block-w 2,6,9,18 --subs 4x1,2x2 --out-blocks 16x5,32x5 --reps 5 --no-l1-acc
echo "=== done $(date -u +%FT%TZ)"
