#!/bin/bash
set -u
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
R=/home/ttuser/agobeljic/tt-metal
cd $R/tt-train
export TT_METAL_HOME=$R TT_METAL_RUNTIME_ROOT=$R
unset TT_MESH_GRAPH_DESC_PATH
PY=../python_env/bin/python
run1d() { rm -f /dev/shm/sm_segment.* 2>/dev/null; echo "=== $1 $(date -u +%FT%TZ)"; timeout 2400 $PY $O/iter34_matmul_sweep1d.py "${@:2}" > $O/iter34_sweep_$1.log 2>&1; echo "rc=$?"; grep -E "ttnn's own|configs tried" $O/iter34_sweep_$1.log | head -3; grep -E "^ +[0-9.]+ ms" $O/iter34_sweep_$1.log | sort -n | head -3; }
run2d() { rm -f /dev/shm/sm_segment.* 2>/dev/null; echo "=== $1 $(date -u +%FT%TZ)"; timeout 2400 $PY $O/iter34_matmul_sweep2d.py "${@:2}" > $O/iter34_sweep_$1.log 2>&1; echo "rc=$?"; grep -E "default:|best|^ *[0-9]+ us" $O/iter34_sweep_$1.log | head -8; }
run1d gateup_fwd_ref11x10 --m 1024 --k 5184 --n 16384 --ref-grid 11x10 --grids 11x10,10x10 --in0-block-w 2,3,4,6 --subs 4x1,2x2 --reps 5
run2d gateup_dx_2d_half 5184 16384 1024 --tb --quick
run2d gateup_dx_2d_full 5184 16384 1024 --tb --quick --full-sync
run2d w2_dx_2d_half 8256 5120 1024 --tb --quick
run2d w2_dx_2d_full 8256 5120 1024 --tb --quick --full-sync
run2d w2_fwd_2d_full 1024 8256 5120 --quick --full-sync
run2d gateup_fwd_2d_full 1024 5184 16384 --quick --full-sync
echo "=== done $(date -u +%FT%TZ)"
