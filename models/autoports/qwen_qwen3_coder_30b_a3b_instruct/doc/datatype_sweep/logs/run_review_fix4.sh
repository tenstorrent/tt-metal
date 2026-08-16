#!/usr/bin/env bash
# Stage-07 review fixes, device phase 4 -- re-measure and re-gate.
#
#   R21  -- norm_fidelity was NOT threaded into decode_residual_norm before this
#           review: the norm built its compute config from the module default and
#           never saw self.precision, so R21_norm_hifi2's original +0.35% was the
#           baseline measured twice, not a HiFi2 norm. Now that the field reaches
#           the op (verified in the audit as norm_math_fidelity), the row is
#           measured for the first time.
#   proof -- selection_proof now checks 21 fields instead of 16 and generates
#           four tokens first, so logits/sampling dtype come off real tensors.
#   gates -- both readiness gates, because tt/ changed.
#   suite -- the full test suite.
set -u
cd /home/raahem/tt-metal
source python_env/bin/activate
unset QWEN3_PRECISION_CONFIG
D=models/autoports/qwen_qwen3_coder_30b_a3b_instruct
L=$D/doc/datatype_sweep/logs
G="$(git rev-parse --short HEAD) + uncommitted stage-07 sweep + review fixes"
hdr () { printf '# cmd: %s\n# git: %s\n# date: %s\n# hw: 1x4 Blackhole P300_X2, FABRIC_1D_RING\n# note: NO precision argument, QWEN3_PRECISION_CONFIG unset -- the default path\n' "$1" "$G" "$(date -Is)"; }

C="python $D/doc/datatype_sweep/probes/sweep_runner.py --force --only R21_norm_hifi2"
hdr "$C" > $L/sweep_review_r21.log; eval $C >> $L/sweep_review_r21.log 2>&1
echo "=== R21 exit $? ==="

C="python $D/doc/datatype_sweep/probes/selection_proof.py"
hdr "$C" > $L/selection_proof.log; eval $C >> $L/selection_proof.log 2>&1
echo "=== selection_proof exit $? ==="

for R in run_teacher_forcing run_prefill_check; do
  C="python -m models.common.readiness_check.$R --model-dir $D --reference $D/readiness_aime24_chat.refpt --mesh-device P300X2 --fabric-config FABRIC_1D_RING --trace-region-size 300000000"
  hdr "$C" > $L/${R}_selected.log; eval $C >> $L/${R}_selected.log 2>&1
  echo "=== $R exit $? ==="
done

C="pytest $D/tests/ -m 'not models_performance_bare_metal' -q"
hdr "$C" > $L/pytest_selected.log
python -m pytest $D/tests/ -m "not models_performance_bare_metal" -q >> $L/pytest_selected.log 2>&1
echo "=== pytest exit $? ==="
