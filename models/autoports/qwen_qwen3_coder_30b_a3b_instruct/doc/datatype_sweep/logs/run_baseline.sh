#!/usr/bin/env bash
# Stage-07 baseline refresh through the new default (PrecisionConfig) construction path.
# One device job at a time, sequentially.
set -u
cd /home/raahem/tt-metal
source python_env/bin/activate
D=models/autoports/qwen_qwen3_coder_30b_a3b_instruct
L=$D/doc/datatype_sweep/logs
GIT="$(git rev-parse --short HEAD) + uncommitted stage-07 precision plumbing"

hdr () { printf '# cmd: %s\n# git: %s\n# date: %s\n# note: default PrecisionConfig (DEFAULT_PRECISION); no precision argument passed\n' "$1" "$GIT" "$(date -Is)"; }

for RUNNER in run_prefill_check run_teacher_forcing; do
  CMD="python -m models.common.readiness_check.$RUNNER --model-dir $D --reference $D/readiness_aime24_chat.refpt --mesh-device P300X2 --fabric-config FABRIC_1D_RING --trace-region-size 300000000"
  hdr "$CMD" > $L/$RUNNER.log
  eval $CMD >> $L/$RUNNER.log 2>&1
  echo "=== $RUNNER exit $? ==="
done

CMD="python $D/doc/datatype_sweep/probes/perf_full_model.py --layers 48 --prompt-len 128 --gen-len 128 --context 8192"
hdr "$CMD" > $L/perf_full_model.log
eval $CMD >> $L/perf_full_model.log 2>&1
echo "=== perf exit $? ==="
