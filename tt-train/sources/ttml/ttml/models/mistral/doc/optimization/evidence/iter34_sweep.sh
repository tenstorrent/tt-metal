#!/bin/bash
# Iteration 34: the base matmuls with dst_full_sync_en (8 fp32 DEST tiles: subblocks up to 8) at the TP shard shapes.
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
  timeout 2400 $PY $O/iter34_matmul_sweep1d.py "${@:2}" > $O/iter34_sweep_$1.log 2>&1; echo "rc=$?"
  grep -E "ttnn's own|^ *[0-9.]+ ms|best|FAIL|mismatch|Error" $O/iter34_sweep_$1.log | head -14
}
# gate_up forward [1024, 5184] x [5184, 16384] (TP shard of 65536): the committed config is 10x10 / w 2 / 4x1 / 16x7
run gateup_fwd_half --m 1024 --k 5184 --n 16384 --grids 10x10,11x10 --in0-block-w 2,3,4,6 --subs 4x1,2x2 --reps 5
run gateup_fwd_full --m 1024 --k 5184 --n 16384 --grids 10x10,11x10 --in0-block-w 2,3,4,6 --subs 4x1,2x2,4x2,2x4,8x1 --full-sync --reps 5
# gate_up dX [5184, 16384] x [1024, 16384]^T
run gateup_dx_half --m 5184 --k 16384 --n 1024 --transpose-b --grids 11x10,10x10 --in0-block-w 2,4,8 --subs 4x1,2x2 --reps 5
run gateup_dx_full --m 5184 --k 16384 --n 1024 --transpose-b --grids 11x10,10x10 --in0-block-w 2,4,8 --subs 4x1,2x2,4x2,2x4,8x1 --full-sync --reps 5
# w2 dX [8256, 5120] x [1024, 5120]^T
run w2_dx_half --m 8256 --k 5120 --n 1024 --transpose-b --grids 11x10,10x10 --in0-block-w 2,4,5,8 --subs 4x1,2x2 --reps 5
run w2_dx_full --m 8256 --k 5120 --n 1024 --transpose-b --grids 11x10,10x10 --in0-block-w 2,4,5,8 --subs 4x1,2x2,4x2,2x4,8x1 --full-sync --reps 5
echo "=== done $(date -u +%FT%TZ)"
