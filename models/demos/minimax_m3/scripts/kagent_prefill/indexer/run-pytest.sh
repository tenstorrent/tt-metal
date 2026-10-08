#!/bin/bash
# usage: run-pytest.sh <tag> <timeout_s> [VAR=val ...] -- <pytest args>  -- pytest on the prefill partition (never
# run_safe_pytest.sh). BASE=1 among the VAR=val runs against the unmodified indexer kernels (base-root).
set -u
tag=$1; to=$2; shift 2
source /mnt/data/kernel-agent/dev/prefill-indexer/env.sh >/dev/null 2>&1
envs=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
  if [ "$1" = "BASE=1" ]; then envs+=(TT_METAL_HOME=$KP/base-root TT_METAL_RUNTIME_ROOT=$KP/base-root TT_METAL_CACHE=$KP/jit-cache-base)
  else envs+=("$1"); fi; shift
done
shift
mkdir -p $KP/runs/$tag
exec /mnt/data/kernel-agent/bin/tt-partition-run prefill --timeout $to --log $KP/runs/$tag/log.txt -- \
  env TT_METAL_OPERATION_TIMEOUT_SECONDS=${TT_METAL_OPERATION_TIMEOUT_SECONDS:-30} "${envs[@]}" \
  python -m pytest -p no:cacheprovider -q -x "$@"
