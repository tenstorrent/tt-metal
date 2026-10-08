#!/bin/bash
# usage: run-bench.sh <tag> <timeout_s> [VAR=val ...]   -- runs kagent_prefill_bench.py on the prefill partition
set -u
tag=$1; to=$2; shift 2
source /mnt/data/kernel-agent/dev/prefill/env.sh >/dev/null 2>&1
mkdir -p $KP/runs/$tag
exec /mnt/data/kernel-agent/bin/tt-partition-run prefill --timeout $to --log $KP/runs/$tag/log.txt -- \
  env TT_METAL_OPERATION_TIMEOUT_SECONDS=${TT_METAL_OPERATION_TIMEOUT_SECONDS:-30} \
      BENCH_TOKENS=${BENCH_TOKENS:-$KP/prompts/natural_56320} BENCH_RESULTS_JSONL=$KP/runs/results.jsonl \
      BENCH_TAG=$tag BENCH_STOP_FILE=$KP/runs/$tag/STOP "$@" \
      python -u models/demos/minimax_m3/tests/perf/kagent_prefill_bench.py
