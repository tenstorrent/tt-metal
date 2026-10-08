#!/bin/bash
# usage: run-bench-old.sh <tag> <timeout_s> [VAR=val ...] -- the bench on the UNMODIFIED kagent/m3-prefill tree
# (dev/prefill/tt-metal, old indexer kernels: the JIT resolves kernel sources relative to the cwd), own JIT cache.
set -u
tag=$1; to=$2; shift 2
source /mnt/data/kernel-agent/dev/prefill-best/env.sh >/dev/null 2>&1
O=/mnt/data/kernel-agent/dev/prefill/tt-metal
export TT_METAL_HOME=$O TT_METAL_RUNTIME_ROOT=$O PYTHONPATH=$O:$O/ttnn:$O/tools TT_METAL_CACHE=$KP/jit-cache-old
cd $O
mkdir -p $KP/runs/$tag
exec /mnt/data/kernel-agent/bin/tt-partition-run prefill --timeout $to --log $KP/runs/$tag/log.txt -- \
  env TT_METAL_OPERATION_TIMEOUT_SECONDS=${TT_METAL_OPERATION_TIMEOUT_SECONDS:-30} \
      BENCH_TOKENS=${BENCH_TOKENS:-$KP/prompts/natural_56320} BENCH_RESULTS_JSONL=$KP/runs/results.jsonl \
      BENCH_TAG=$tag BENCH_STOP_FILE=$KP/runs/$tag/STOP "$@" \
      python -u models/demos/minimax_m3/tests/perf/kagent_prefill_bench.py
