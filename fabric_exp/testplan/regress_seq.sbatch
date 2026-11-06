#!/bin/bash
# Several `run_group.py job` runs back to back in ONE allocation (e.g. a same-node A/B across two trees). Each line of
# $REGRESS_SEQ is: STAGE_DIR|MANIFEST|TREE_NAME|TREE_PATH|GROUP|LABEL|ARGS (ARGS space-separated). Each job resets the
# device and checks it at its start, so arms are separated by a forced reset; the first job on a node runs the node
# check. A job's non-zero exit does not stop the sequence (both arms always run), except 75 (node busy) and 76 (node
# check failed), which stop it.
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=/data/aagarwal/regress/logs/%x-%j.log
#SBATCH --chdir=/data/aagarwal/regress
set -uo pipefail
SEQ=${REGRESS_SEQ:?REGRESS_SEQ not set}
echo "regress sequence job ${SLURM_JOB_ID:-} on $(hostname) start $(date -u +%FT%TZ): $(grep -c . $SEQ) arm(s)"
unset $(env | grep -oE '^TT_[A-Za-z0-9_]*' | tr '\n' ' ') PYTHONPATH VIRTUAL_ENV 2>/dev/null
n=0; rcs=""
while IFS='|' read -r ST MAN TREE TREE_PATH GROUP LABEL ARGS; do
    [ -z "$ST" ] && continue
    n=$((n + 1))
    echo "=== arm $n: $GROUP ($TREE, $LABEL) $(date -u +%T): $ARGS"
    PY=$TREE_PATH/python_env/bin/python
    $PY -c 'import yaml' > /dev/null 2>&1 || PY=python3
    # shellcheck disable=SC2086
    $PY -u $ST/run_group.py --manifest $MAN job --group "$GROUP" --tree "$TREE" --label "$LABEL" $ARGS < /dev/null
    rc=$?
    rcs="$rcs arm$n=$rc"
    echo "=== arm $n done rc=$rc $(date -u +%T)"
    [ $rc = 75 ] || [ $rc = 76 ] && { echo "!!! stopping the sequence (node busy / node check)"; break; }
done < $SEQ
echo "sequence done:$rcs $(date -u +%T)"
