#!/usr/bin/env bash
# Re-print every table from an EXISTING campaign. Pure post-processing -- touches no devices,
# so it is safe to run while someone else has the galaxy.
#
#   ./make_tables.sh                 # the 2026-09-14 campaign
#   RES=<dir> PROF=<dir> ./make_tables.sh
set -u
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # .../pipeline_prefill_harness/wrappers
HARNESS="$(cd "$S/.." && pwd)"
cd "$(cd "$HARNESS/../../../../../.." && pwd)" || exit 1    # repo root
T=models/demos/deepseek_v3_d_p/tests/perf
source "$HARNESS/env.sh"
H=$(hostname)
RES=${RES:-$PWD/mistral4_perf_$H}
# Default to the 2026-09-14 09:xx re-run captures; the originals are in mistral4_perf_profile_$H.
PROF=${PROF:-$PWD/mistral4_perf_profile_$H}

echo "############ campaign summary (throughput + latency + layer budget) ############"
echo "results: $RES"; echo "profile: $PROF"; echo
"$PY" $T/summarize_prefill_campaign.py "$RES" "$PROF"

echo; echo "############ per-cell throughput detail (warmup 8) ############"
for cfg in 1rank pp4; do for isl in 5120 25600 102400 261120; do
  L="$RES/${cfg}_${isl}_thru/runner.log"
  [ -s "$L" ] && { echo "---- ${cfg}_${isl}_thru ----"; "$PY" $T/analyze_prefill_throughput.py "$L" 8 2>&1 | tail -3; } \
               || echo "---- ${cfg}_${isl}_thru: NO LOG ----"
done; done

echo; echo "############ per-layer budget + stage asymmetry ############"
for spec in "1rank_deep 0" "pp4_deep 0" "pp4_deep 1" "pp4_deep 2" "pp4_deep 3"; do
  set -- $spec
  csv=$(ls -1 "$PROF/$1/rank$2"/reports/*/*/ops_perf_results*.csv 2>/dev/null | tail -1)
  [ -n "$csv" ] && "$PY" $T/analyze_prefill_layer_budget.py "$csv" "$1 rank$2" || echo "-- $1 rank$2: NO CSV"
done

echo; echo "############ KV-depth ramp ############"
for spec in "1rank_deep 0" "pp4_deep 0"; do
  set -- $spec
  csv=$(ls -1 "$PROF/$1/rank$2"/reports/*/*/ops_perf_results*.csv 2>/dev/null | tail -1)
  [ -n "$csv" ] && "$PY" $T/analyze_prefill_kv_ramp.py "$csv" "$1 rank$2"
done

echo; echo "############ integrity: trap 1 -- rc=0 can hide a dead runner ############"
for d in "$RES"/*/; do
  [ -f "$d/runner.log" ] || continue
  n=$(grep -cE "AssertionError|Traceback|TT_FATAL" "$d/runner.log" 2>/dev/null || echo 0)
  [ "$n" != "0" ] && echo "  DIRTY $(basename $d): $n error lines" || echo "  clean $(basename $d)"
done
