#!/usr/bin/env bash
# Slurm array task: map $SLURM_ARRAY_TASK_ID -> op (that line of $OPS_LIST),
# run that one op to completion, exit (frees the galaxy). Slurm is the scheduler, queue and
# refill: submit the whole set at once and it runs as many as there are idle galaxies,
# queues the rest, and --requeue retries a died task. No supervisor, no passes, no waits.
#
# Submit (one line):
#   sbatch --array=1-<N> --requeue --export=ALL -J run_op \
#          -p <glx-partitions> --exclude=<poisoned nodes> --time=720 run_op_array.sh
# with env exported: OPS_LIST (one op per line) and run_op.sh's own:
# SWEEP OPS_TSV IDMAP BUILD VENV LLK_HOME PYDIR OUT.
set -uo pipefail
op=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${OPS_LIST:?}")
[ -n "$op" ] || { echo "no op at array index $SLURM_ARRAY_TASK_ID"; exit 1; }
exec bash "$(dirname "$0")/run_op.sh" "$op"
