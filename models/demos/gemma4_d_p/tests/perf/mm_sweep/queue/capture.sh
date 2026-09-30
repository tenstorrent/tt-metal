#!/bin/bash
# capture.sh <label> <chunk_idx> <chunk_size> [ENV=V ...]: tracy layer capture (both layer types) + tt-perf-report tables.
source /data/kmabee/runs_sasha/env.sh
L=$1; N=$2; C=$3; shift 3
export PATH=$W/python_env/bin:$PATH TT_METAL_HOME=$W PYTHONPATH=$W/ttnn:$W PYTEST_TIMEOUT=7200
OUT=$O/prof/${L}_c${C}_chunk$N; rm -rf $OUT; mkdir -p $OUT
waitchips
echo "=== capture $L c$C chunk$N start $(date +%T) sha=$(git -C $W rev-parse --short HEAD) env=$* tdp=$(tdp)"
cd $W && env "$@" TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000 timeout 5400 $PY -m tracy -r -p -v -o $OUT/profiler -m pytest \
  "models/demos/gemma4_d_p/demo/text_demo_prefill.py::test_prefill_layer_perf_chunk_n[blackhole-chunk$N-both-sz$C-ctx_256k-8x4]" \
  -sv -p no:cacheprovider < /dev/null > $OUT/run.log 2>&1
echo "rc=$? $(date +%T) $(grep -E '[0-9]+ (passed|failed)' $OUT/run.log | tail -1 | tr -d '=')"
for i in $(seq 1 3); do CSV=$(find $OUT/profiler -name 'ops_perf_results_*.csv' -size +1M 2>/dev/null | head -1); [ -n "$CSV" ] && break; sleep 20; done
for T in global local; do
  tt-perf-report --no-color --start-signpost gemma4-layer-$T-chunk$N-start --end-signpost gemma4-layer-$T-chunk$N-stop "$CSV" > $OUT/$T.txt 2>&1
  tt-perf-report --start-signpost gemma4-layer-$T-chunk$N-start --end-signpost gemma4-layer-$T-chunk$N-stop --csv $OUT/$T.csv "$CSV" > /dev/null 2>&1
done
[ -s "$CSV" ] && cp "$CSV" $OUT/ops_perf_results.csv && [ -s $OUT/ops_perf_results.csv ] && rm -rf $OUT/profiler
echo "=== done $L c$C $(date +%T)"
