#!/bin/bash
# usage: run-op.sh <tag> <timeout_s> [args to ${OP_BENCH:-kagent_msa_op_bench.py}]  -- single-chip op PCC + device time (OP_BENCH=kagent_indexer_op_bench.py for the indexer)
set -u
tag=$1; to=$2; shift 2
source /mnt/data/kernel-agent/dev/prefill/env.sh >/dev/null 2>&1
mkdir -p $KP/runs/$tag
exec /mnt/data/kernel-agent/bin/tt-partition-run prefill --timeout $to --log $KP/runs/$tag/log.txt -- \
  env TT_METAL_OPERATION_TIMEOUT_SECONDS=${TT_METAL_OPERATION_TIMEOUT_SECONDS:-30} LOGURU_LEVEL=INFO \
  python -u models/demos/minimax_m3/tests/perf/${OP_BENCH:-kagent_msa_op_bench.py} "$@"
