#!/usr/bin/env bash
set -u
cd /home/raahem/tt-metal
source python_env/bin/activate
unset QWEN3_PRECISION_CONFIG
D=models/autoports/qwen_qwen3_coder_30b_a3b_instruct
L=$D/doc/datatype_sweep/logs
C="python $D/doc/datatype_sweep/probes/selection_proof.py"
printf '# cmd: %s\n# git: %s + uncommitted stage-07 sweep\n# date: %s\n# hw: 1x4 Blackhole P300_X2, FABRIC_1D_RING\n# note: NO precision argument, QWEN3_PRECISION_CONFIG unset\n' "$C" "$(git rev-parse --short HEAD)" "$(date -Is)" > $L/selection_proof.log
eval $C >> $L/selection_proof.log 2>&1
echo "=== selection_proof exit $? ==="
