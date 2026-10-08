#!/bin/bash
# usage: run-ops.sh <tag> <timeout_s> <cmdfile>  -- run several single-chip op-bench lines inside ONE prefill-partition
# job. Each non-comment line of <cmdfile>: "<label> [VAR=val ...] <script.py> <args...>" (script under
# models/demos/minimax_m3/tests/perf/, run from the worktree). BASE=1 runs it against the unmodified indexer kernels
# (base-root symlink tree, own JIT cache).
set -u
tag=$1; to=$2; cmdfile=$(realpath "$3")
source /mnt/data/kernel-agent/dev/prefill-indexer/env.sh >/dev/null 2>&1
mkdir -p $KP/runs/$tag
cp $cmdfile $KP/runs/$tag/cmds.txt
exec /mnt/data/kernel-agent/bin/tt-partition-run prefill --timeout $to --log $KP/runs/$tag/log.txt -- \
  env TT_METAL_OPERATION_TIMEOUT_SECONDS=${TT_METAL_OPERATION_TIMEOUT_SECONDS:-30} LOGURU_LEVEL=INFO \
  bash $KP/run-ops-inner.sh $KP/runs/$tag/cmds.txt
