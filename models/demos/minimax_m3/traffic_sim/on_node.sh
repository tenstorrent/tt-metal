#!/bin/bash
# Run a command on a Slurm compute node with the soft ulimits raised to the hard limits
# (exabox compute nodes default to soft -u 512 threads / -t 24 CPU-h, which kills long multi-process runs).
# Usage: JOB=<slurm job id> ./on_node.sh <cmd> [args...]     (without JOB it runs locally)
set -e
if [ -n "$JOB" ] && [ -z "$SLURM_STEP_ID" ]; then
  exec srun --jobid "$JOB" --overlap "$0" "$@"
fi
ulimit -u "$(ulimit -Hu)" 2>/dev/null || true
ulimit -t unlimited 2>/dev/null || true
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
ulimit -s "$(ulimit -Hs)" 2>/dev/null || true
exec "$@"
