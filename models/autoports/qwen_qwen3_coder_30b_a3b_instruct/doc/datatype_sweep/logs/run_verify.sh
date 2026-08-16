#!/usr/bin/env bash
set -u
cd /home/raahem/tt-metal
source python_env/bin/activate
unset QWEN3_PRECISION_CONFIG
D=models/autoports/qwen_qwen3_coder_30b_a3b_instruct
L=$D/doc/datatype_sweep/logs
printf '# cmd: pytest tests/ -m "not models_performance_bare_metal" -q (full suite, after kv_cache_dtype audit spelling fix)\n# git: %s + uncommitted stage-07 sweep\n# date: %s\n# note: NO precision argument -- the default path, which is now the selected config\n' "$(git rev-parse --short HEAD)" "$(date -Is)" > $L/pytest_selected.log
python -m pytest $D/tests/ -m "not models_performance_bare_metal" -q >> $L/pytest_selected.log 2>&1
echo "=== pytest exit $? ==="
C="python $D/doc/datatype_sweep/probes/selection_proof.py"
printf '# cmd: %s\n# git: %s\n# date: %s\n' "$C" "$(git rev-parse --short HEAD)" "$(date -Is)" > $L/selection_proof.log
eval $C >> $L/selection_proof.log 2>&1
echo "=== selection_proof exit $? ==="
