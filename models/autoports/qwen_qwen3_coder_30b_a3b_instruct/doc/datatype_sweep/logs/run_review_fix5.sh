#!/usr/bin/env bash
# Stage-07 re-review fix 5: re-run the post-selection token-out benchmark on the
# CURRENT tree. The published perf_full_model_selected.json was measured at
# 07:50, before the review fixes landed in tt/ at 09:23+, so its headline no
# longer described the shipped tree. Same command as run_final_stage07.sh --
# the normal selected-config construction path, no precision argument.
set -u
cd /home/raahem/tt-metal
source python_env/bin/activate
unset QWEN3_PRECISION_CONFIG
D=models/autoports/qwen_qwen3_coder_30b_a3b_instruct
L=$D/doc/datatype_sweep/logs
G="$(git rev-parse --short HEAD) + uncommitted stage-07 sweep + re-review fixes (selected config is the default)"
hdr () { printf '# cmd: %s\n# git: %s\n# date: %s\n# hw: 1x4 Blackhole P300_X2, FABRIC_1D_RING\n# note: NO precision argument, QWEN3_PRECISION_CONFIG unset -- the default path\n' "$1" "$G" "$(date -Is)"; }

C="python $D/doc/datatype_sweep/probes/perf_full_model.py --layers 48 --prompt-len 128 --gen-len 128 --context 8192 --tag _selected"
hdr "$C" > $L/perf_full_model_selected.log; eval $C >> $L/perf_full_model_selected.log 2>&1
echo "=== perf exit $? ==="
