#!/bin/bash
# usage: run-prof.sh <tag> <timeout_s> [VAR=val ...] -- device-profiler (tracy) capture of the bench's request on a
# few layers (BENCH_LAYER_IDS), then the zone roll-up. Post-process RSS ~ 7 GB per profiled layer (16 chips); keep <= 8.
set -u
tag=$1; to=$2; shift 2
source /mnt/data/kernel-agent/dev/prefill/env.sh >/dev/null 2>&1
out=$KP/runs/$tag; mkdir -p $out
/mnt/data/kernel-agent/bin/tt-partition-run prefill --timeout $to --log $out/log.txt -- \
  env TT_METAL_OPERATION_TIMEOUT_SECONDS=${TT_METAL_OPERATION_TIMEOUT_SECONDS:-30} \
      BENCH_TOKENS=${BENCH_TOKENS:-$KP/prompts/natural_56320} BENCH_TAG=$tag BENCH_STOP_FILE=$out/STOP BENCH_PROFILE=1 "$@" \
      python -m tracy -r -p -v --no-web-server -o $out/tracy --op-support-count 4000 \
      models/demos/minimax_m3/tests/perf/kagent_prefill_bench.py
rc=$?
csv=$(ls -t $out/tracy/reports/*/ops_perf_results_*.csv 2>/dev/null | head -1)
if [ -n "$csv" ]; then
  cp "$csv" $out/ops.csv
  python models/demos/minimax_m3/tests/perf/parse_zone_perf.py "$csv" --json $out/zones.json --top 5 > $out/zones.txt 2>&1
  python models/demos/minimax_m3/tests/perf/kagent_ops_breakdown.py "$csv" --json $out/ops.json > $out/ops.txt 2>&1
  # the intermediates are large and worthless once the ops CSV exists
  rm -rf $out/tracy/.logs $out/tracy/reports/*/profile_log_device.csv $out/tracy/reports/*/*.tracy
fi
exit $rc
