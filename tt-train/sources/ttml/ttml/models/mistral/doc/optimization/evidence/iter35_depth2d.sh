#!/bin/bash
# Iteration 35: 2D K-block depth on the model's grids: the dX^T shapes (full grid) and the w2 forward at its
# committed 11x8 config, with K padded 8256 -> 8320 (260 = 4 x 5 x 13 tiles: depths 10 / 13 / 20 / 26 possible).
set -u
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
R=/home/ttuser/agobeljic/tt-metal
cd $R/tt-train
export TT_METAL_HOME=$R TT_METAL_RUNTIME_ROOT=$R
unset TT_MESH_GRAPH_DESC_PATH TT_METAL_CACHE
PY=$R/python_env/bin/python
run() { rm -f /dev/shm/sm_segment.* 2>/dev/null; echo "=== $1 $(date -u +%FT%TZ)"; timeout 900 $PY $O/iter35_dx_depth.py "${@:2}" 2>&1 | grep -E "^M=|^  2D|FAILED|not divisible"; }
run gateup_dx --m 5184 --k 16384 --n 1024 --depths 4,8,16 --subs 1x3,1x1
run w2_dx --m 8256 --k 5120 --n 1024 --depths 4,5,8,10,16,20 --subs 1x3,2x1
run w2_fwd_8256 --m 1024 --k 8256 --n 5120 --no-tb --grid 11x8 --depths 3,6 --subs 4x1,2x2
run w2_fwd_8320 --m 1024 --k 8320 --n 5120 --no-tb --grid 11x8 --depths 10,13,20,26 --subs 4x1,2x2
echo "=== done $(date -u +%FT%TZ)"
