#!/usr/bin/env bash
set -u
cd /home/raahem/tt-metal
source python_env/bin/activate
unset QWEN3_PRECISION_CONFIG
D=models/autoports/qwen_qwen3_coder_30b_a3b_instruct
L=$D/doc/datatype_sweep/logs
G="$(git rev-parse --short HEAD) + uncommitted stage-07 sweep (selected config is now the default)"
hdr () { printf '# cmd: %s\n# git: %s\n# date: %s\n# hw: 1x4 Blackhole P300_X2, FABRIC_1D_RING\n# note: NO precision argument, QWEN3_PRECISION_CONFIG unset -- the default path\n' "$1" "$G" "$(date -Is)"; }

C="python $D/doc/datatype_sweep/probes/selection_proof.py"
hdr "$C" > $L/selection_proof.log; eval $C >> $L/selection_proof.log 2>&1
echo "=== selection_proof exit $? ==="

C="python $D/doc/datatype_sweep/probes/perf_full_model.py --layers 48 --prompt-len 128 --gen-len 128 --context 8192 --tag _selected"
hdr "$C" > $L/perf_full_model_selected.log; eval $C >> $L/perf_full_model_selected.log 2>&1
echo "=== perf exit $? ==="

for R in run_teacher_forcing run_prefill_check; do
  C="python -m models.common.readiness_check.$R --model-dir $D --reference $D/readiness_aime24_chat.refpt --mesh-device P300X2 --fabric-config FABRIC_1D_RING --trace-region-size 300000000"
  hdr "$C" > $L/${R}_selected.log; eval $C >> $L/${R}_selected.log 2>&1
  echo "=== $R exit $? ==="
done

C="pytest $D/tests/ -m 'not models_performance_bare_metal' -q"
hdr "$C" > $L/pytest_selected.log
python -m pytest $D/tests/ -m "not models_performance_bare_metal" -q >> $L/pytest_selected.log 2>&1
echo "=== pytest exit $? ==="
