#!/bin/bash
# usage: runjob.sh <tag> <timeout_s> <spec>...   spec = "G:case[:iters[:variant]]" (G=0 legacy; variant = a
# variants/<name> JIT root from mkvariant.py, with its own JIT cache). One tt-partition-run prefill job,
# cases run sequentially; every case compares against runs/base0 (legacy, canonical build) on identical inputs.
set -u
tag=$1; to=$2; shift 2
source /mnt/data/kernel-agent/dev/prefill-sdpa/env.sh >/dev/null 2>&1
out=$KP/runs/$tag; mkdir -p $out
IDX=$KP0/runs/c5120-v2/dump/cuts_ev/msa_block_ids.pt
job=$out/job.sh
{
  echo "set -u"
  echo "export TT_METAL_OPERATION_TIMEOUT_SECONDS=\${TT_METAL_OPERATION_TIMEOUT_SECONDS:-30} LOGURU_LEVEL=INFO"
  for spec in "$@"; do
    IFS=: read G c it var <<< "$spec"; it=${it:-20}
    envp=""; tagv=""
    if [ -n "$var" ]; then
      # cd: the JIT resolves a kernel's relative source path against the cwd before TT_METAL_HOME
      envp="cd $KP/variants/$var && TT_METAL_HOME=$KP/variants/$var TT_METAL_RUNTIME_ROOT=$KP/variants/$var TT_METAL_CACHE=$KP/jit-cache-$var"; tagv="_$var"
    fi
    case $c in
      l30) a="--S 1280 --T 56320 --chunk-start 51200 --rank 0 --indices $IDX --layer 30";;
      l3)  a="--S 1280 --T 56320 --chunk-start 51200 --rank 3 --indices $IDX --layer 3";;
      syn) a="--S 1280 --T 56320 --chunk-start 51200 --rank 0";;
    esac
    echo "echo '=== G=$G case $c ${var:+variant $var} ==='"
    echo "cd $W; $envp TT_MSA_PACKED_GROUP=$G python -u models/demos/minimax_m3/tests/perf/kagent_msa_op_bench.py $a --iters $it --ref-all --compare $KP/runs/base0/out_$c.pt --dump-out $out/out_G${G}_$c$tagv.pt || { echo CASE FAILED; exit 1; }"
  done
} > $job
exec /mnt/data/kernel-agent/bin/tt-partition-run prefill --timeout $to --log $out/log.txt -- bash $job
