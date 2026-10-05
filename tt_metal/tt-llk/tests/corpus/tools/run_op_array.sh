#!/usr/bin/env bash
# Slurm array task: map $SLURM_ARRAY_TASK_ID -> op (that line of $OPS_LIST),
# run that one op to completion, exit (frees the galaxy). Slurm is the scheduler, queue and
# refill: submit the whole set at once and it runs as many as there are idle galaxies,
# queues the rest, and --requeue retries a died task. No supervisor, no passes, no waits.
#
# Submit (one line):
#   sbatch --array=1-<N> --requeue --export=ALL -J run_op \
#          -p <glx-partitions> --exclude=<poisoned nodes> --time=720 run_op_array.sh
# with env exported: OPS_LIST (one op per line) and either:
#   breadth: SWEEP OPS_TSV IDMAP BUILD VENV LLK_HOME PYDIR OUT
#   32-chip: GALAXY_SHARD=1 SWEEP OPS_TSV IDMAP FLAGS_TSV FARM_ROOT VENV OUT
set -uo pipefail
op=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${OPS_LIST:?}")
[ -n "$op" ] || { echo "no op at array index $SLURM_ARRAY_TASK_ID"; exit 1; }
if [ "${GALAXY_SHARD:-0}" = 1 ]; then
  : "${OUT:?} ${FARM_ROOT:?} ${VENV:?}"
  ulimit -u "$(ulimit -Hu)" 2>/dev/null || true
  ulimit -n 131072 2>/dev/null || true
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1 MALLOC_CONF=background_thread:false
  export OP="$op" OUT="$OUT/$op"
  exec bash "$(dirname "$0")/galaxy_shard.sh"
fi
exec bash "$(dirname "$0")/run_op.sh" "$op"
