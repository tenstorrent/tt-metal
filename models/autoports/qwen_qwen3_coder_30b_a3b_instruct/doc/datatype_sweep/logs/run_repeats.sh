#!/usr/bin/env bash
set -u
cd /home/raahem/tt-metal
source python_env/bin/activate
D=models/autoports/qwen_qwen3_coder_30b_a3b_instruct
L=$D/doc/datatype_sweep/logs
C="python $D/doc/datatype_sweep/probes/repeats.py --ids R00_default,R25_gateup64_down24 --n 3"
printf '# cmd: %s\n# git: %s + uncommitted stage-07 sweep\n# date: %s\n# hw: 1x4 Blackhole P300_X2, FABRIC_1D_RING\n# purpose: measure the run-to-run band on the ranking metric so the frontier is not drawn through noise\n' "$C" "$(git rev-parse --short HEAD)" "$(date -Is)" > $L/repeats.log
eval $C >> $L/repeats.log 2>&1
echo "=== repeats exit $? ==="
