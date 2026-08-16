#!/usr/bin/env bash
set -u
cd /home/raahem/tt-metal
source python_env/bin/activate
D=models/autoports/qwen_qwen3_coder_30b_a3b_instruct
L=$D/doc/datatype_sweep/logs
printf '# cmd: pytest %s/tests/ -m "not models_performance_bare_metal" -q\n# git: %s + uncommitted stage-07 precision plumbing (final tree)\n# date: %s\n# note: default PrecisionConfig (DEFAULT_PRECISION); no precision argument passed\n' "$D" "$(git rev-parse --short HEAD)" "$(date -Is)" > $L/pytest_full_suite.log
python -m pytest $D/tests/ -m "not models_performance_bare_metal" -q >> $L/pytest_full_suite.log 2>&1
echo "=== pytest exit $? ==="
