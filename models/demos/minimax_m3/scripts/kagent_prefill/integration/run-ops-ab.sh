#!/bin/bash
# usage: run-ops-ab.sh <tag> <timeout_s> <cmdfile> -- several single-chip op-bench lines in ONE prefill-partition job.
# Each line: "<label> <old|new> [VAR=val ...] <script.py> <args...>". The JIT resolves kernel sources relative to the
# CURRENT DIRECTORY first, so "old" runs cd into the kagent/m3-prefill worktree (unmodified kernels) with its own
# PYTHONPATH / roots and a separate JIT cache; "new" runs in the integration worktree. The harness script is always
# the integration tree's copy (it has --dump).
set -u
tag=$1; to=$2; cmdfile=$(realpath "$3")
source /mnt/data/kernel-agent/dev/prefill-best/env.sh >/dev/null 2>&1
mkdir -p $KP/runs/$tag
cp $cmdfile $KP/runs/$tag/cmds.txt
exec /mnt/data/kernel-agent/bin/tt-partition-run prefill --timeout $to --log $KP/runs/$tag/log.txt -- \
  env TT_METAL_OPERATION_TIMEOUT_SECONDS=${TT_METAL_OPERATION_TIMEOUT_SECONDS:-30} LOGURU_LEVEL=INFO \
  bash $KP/run-ops-ab-inner.sh $KP/runs/$tag/cmds.txt
