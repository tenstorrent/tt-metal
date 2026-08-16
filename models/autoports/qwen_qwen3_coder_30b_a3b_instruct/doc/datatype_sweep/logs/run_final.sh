#!/usr/bin/env bash
set -u
cd /home/raahem/tt-metal
source python_env/bin/activate
D=models/autoports/qwen_qwen3_coder_30b_a3b_instruct
L=$D/doc/datatype_sweep/logs
GIT="$(git rev-parse --short HEAD) + uncommitted stage-07 precision plumbing (final tree)"
hdr () { printf '# cmd: %s\n# git: %s\n# date: %s\n# note: default PrecisionConfig (DEFAULT_PRECISION); no precision argument passed\n' "$1" "$GIT" "$(date -Is)"; }

CMD="python -m models.common.readiness_check.run_prefill_check --model-dir $D --reference $D/readiness_aime24_chat.refpt --mesh-device P300X2 --fabric-config FABRIC_1D_RING --trace-region-size 300000000"
hdr "$CMD" > $L/run_prefill_check.log
eval $CMD >> $L/run_prefill_check.log 2>&1
echo "=== prefill exit $? ==="

CMD="pytest $D/tests/ -m \"not models_performance_bare_metal\" -q"
hdr "$CMD" > $L/pytest_full_suite.log
python -m pytest $D/tests/ -m "not models_performance_bare_metal" -q >> $L/pytest_full_suite.log 2>&1
echo "=== pytest exit $? ==="
