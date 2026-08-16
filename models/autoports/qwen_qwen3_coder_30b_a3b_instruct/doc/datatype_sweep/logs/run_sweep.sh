#!/usr/bin/env bash
set -u
cd /home/raahem/tt-metal
source python_env/bin/activate
D=models/autoports/qwen_qwen3_coder_30b_a3b_instruct
L=$D/doc/datatype_sweep/logs
CMD="python $D/doc/datatype_sweep/probes/sweep_runner.py"
printf '# cmd: %s\n# git: %s + uncommitted stage-07 sweep\n# date: %s\n# hw: 1x4 Blackhole P300_X2, FABRIC_1D_RING\n# regime: TIER B -- full 48-layer traced teacher forcing, one subprocess per row\n' "$CMD" "$(git rev-parse --short HEAD)" "$(date -Is)" > $L/sweep_tierB.log
eval $CMD >> $L/sweep_tierB.log 2>&1
echo "=== sweep exit $? ==="
