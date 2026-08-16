#!/usr/bin/env bash
set -u
cd /home/raahem/tt-metal
source python_env/bin/activate
D=models/autoports/qwen_qwen3_coder_30b_a3b_instruct
L=$D/doc/datatype_sweep/logs
CMD="python $D/doc/datatype_sweep/probes/structural_probe.py --layers 2"
printf '# cmd: %s\n# git: %s + uncommitted stage-07 sweep\n# date: %s\n# hw: 1x4 Blackhole P300_X2, FABRIC_1D_RING\n# note: TIER A structural only -- 2 layers, no accuracy/perf numbers valid here\n' "$CMD" "$(git rev-parse --short HEAD)" "$(date -Is)" > $L/structural_probe.log
eval $CMD >> $L/structural_probe.log 2>&1
echo "=== structural exit $? ==="
