#!/usr/bin/env bash
# laneMK Slurm array task: map $SLURM_ARRAY_TASK_ID -> op (that line of $LANEMK_OPS_LIST),
# run that one op to completion, exit (frees the galaxy). Slurm is the scheduler, queue and
# refill: submit the whole set at once and it runs as many as there are idle galaxies,
# queues the rest, and --requeue retries a died task. No supervisor, no passes, no waits.
#
# Submit (one line):
#   sbatch --array=1-<N> --requeue --export=ALL -J lanemk_op \
#          -p <glx-partitions> --exclude=<poisoned nodes> --time=720 lanemk_array.sh
# with env exported: LANEMK_OPS_LIST (one op per line) LANEMK_RUN_OP (path to
# lanemk_run_op.sh) and run_op's own: OPS_TSV IDMAP BUILD VENV LLK_HOME PYDIR OUT.
set -uo pipefail
op=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${LANEMK_OPS_LIST:?}")
[ -n "$op" ] || { echo "no op at array index $SLURM_ARRAY_TASK_ID"; exit 1; }
exec bash "${LANEMK_RUN_OP:?}" "$op"
