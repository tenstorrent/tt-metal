#!/bin/bash
# One regression group on one Slurm node (Galaxy 110-a or an approved extra LoudBox). Never sbatch this by hand:
# `run_group.py submit` stages it with the runner and manifest under /data/aagarwal/regress/stage/<stamp>-<group>/
# and passes partition, node, time and the REGRESS_* variables. The work is `run_group.py job`: node guard, busy
# check, per-job TT_METAL_HOME farm on node-local /tmp, reset + test_system_health, the steps of every router / pass,
# results to /data/aagarwal/regress/results/<label>/<group>/.
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=/data/aagarwal/regress/logs/%x-%j.log
#SBATCH --chdir=/data/aagarwal/regress
set -uo pipefail
ST=${REGRESS_STAGE:?REGRESS_STAGE not set}
TREE=${REGRESS_TREE_PATH:?REGRESS_TREE_PATH not set}
echo "regress job ${SLURM_JOB_ID:-} on $(hostname) start $(date -u +%FT%TZ) group ${REGRESS_GROUP:-?} tree $TREE label ${REGRESS_LABEL:-?}"
# the submitting shell's TT_* / venv settings must not reach the steps (the runner sets them per step)
unset $(env | grep -oE '^TT_[A-Za-z0-9_]*' | tr '\n' ' ') PYTHONPATH VIRTUAL_ENV 2>/dev/null
PY=$TREE/python_env/bin/python
if ! $PY -c 'import yaml' > /dev/null 2>&1; then
    echo "the tree's venv python does not run here; falling back to python3"
    PY=python3
fi
ARGS=${REGRESS_ARGS:-}
[ -f $ST/args.txt ] && ARGS=$(cat $ST/args.txt)
# shellcheck disable=SC2086
exec $PY -u $ST/run_group.py --manifest $ST/manifest.yaml job --group "$REGRESS_GROUP" --tree "$REGRESS_TREE" \
    --label "$REGRESS_LABEL" $ARGS
