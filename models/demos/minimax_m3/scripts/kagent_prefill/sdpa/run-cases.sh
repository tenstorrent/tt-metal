#!/bin/bash
# usage: run-cases.sh <tag> <timeout_s> [--iters N] [--no-dump] [--compare-tag BASE] [--cases "l30 l3 syn"] [--extra "args"]
# Runs kagent_msa_op_bench.py for the three gate index sets (L30 rank 0, L3 rank 3, synthetic) in ONE
# tt-partition-run prefill job (sequential processes). Env overrides honoured: MSA_ROOT (TT_METAL_HOME/RUNTIME_ROOT
# of a variant tree), MSA_JIT (its JIT cache).
set -u
tag=$1; to=$2; shift 2
iters=20; dump=1; cmp=""; cases="l30 l3 syn"; extra=""
while [ $# -gt 0 ]; do case $1 in
  --iters) iters=$2; shift 2;; --no-dump) dump=0; shift;; --compare-tag) cmp=$2; shift 2;;
  --cases) cases=$2; shift 2;; --extra) extra=$2; shift 2;; *) echo "bad arg $1"; exit 2;; esac; done
source /mnt/data/kernel-agent/dev/prefill-sdpa/env.sh >/dev/null 2>&1
out=$KP/runs/$tag; mkdir -p $out
IDX=$KP0/runs/c5120-v2/dump/cuts_ev/msa_block_ids.pt
job=$out/job.sh
{
  echo "set -u"
  echo "export TT_METAL_OPERATION_TIMEOUT_SECONDS=\${TT_METAL_OPERATION_TIMEOUT_SECONDS:-30} LOGURU_LEVEL=INFO"
  [ -n "${MSA_ROOT:-}" ] && echo "export TT_METAL_HOME=$MSA_ROOT TT_METAL_RUNTIME_ROOT=$MSA_ROOT"
  [ -n "${MSA_JIT:-}" ] && echo "export TT_METAL_CACHE=$MSA_JIT"
  echo "rc=0"
  for c in $cases; do
    case $c in
      l30) a="--S 1280 --T 56320 --chunk-start 51200 --rank 0 --indices $IDX --layer 30";;
      l3)  a="--S 1280 --T 56320 --chunk-start 51200 --rank 3 --indices $IDX --layer 3";;
      syn) a="--S 1280 --T 56320 --chunk-start 51200 --rank 0";;
      l59) a="--S 1280 --T 56320 --chunk-start 51200 --rank 1 --indices $IDX --layer 59";;
    esac
    d=""; [ $dump = 1 ] && d="--dump-out $out/out_$c.pt"
    k=""; [ -n "$cmp" ] && k="--compare $KP/runs/$cmp/out_$c.pt"
    echo "echo '=== case $c ==='"
    echo "python -u models/demos/minimax_m3/tests/perf/kagent_msa_op_bench.py $a --iters $iters --ref-all $d $k $extra || rc=\$?"
    echo "[ \$rc -ne 0 ] && { echo 'CASE FAILED rc='\$rc; exit \$rc; }"
  done
  echo "exit \$rc"
} > $job
exec /mnt/data/kernel-agent/bin/tt-partition-run prefill --timeout $to --log $out/log.txt -- bash $job
