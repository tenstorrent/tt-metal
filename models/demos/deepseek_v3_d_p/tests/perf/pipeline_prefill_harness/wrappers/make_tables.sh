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
# `grep -c` exits 1 when the count is zero, so the old `|| echo 0` appended a SECOND "0" and made
# $n the two-line string "0\n0", which is != "0" -- every cell reported DIRTY and a real failure
# was indistinguishable from a clean run. grep -c always prints a count, so no fallback is needed.
#
# TT_FATAL is also split out: rank teardown legitimately logs `TT_FATAL: cq_id 0 is out of range`
# from the D2D stream-service destructors after the device closes, on every healthy run. Counting it
# fails every good cell; ignoring it wholesale hides real ones. Both counts are printed.
for d in "$RES"/*/; do
  [ -f "$d/runner.log" ] || continue
  py=$(grep -cE "AssertionError|Traceback \(most recent call last\)" "$d/runner.log" 2>/dev/null)
  tf=$(grep -E "TT_FATAL" "$d/runner.log" 2>/dev/null | grep -cvE "cq_id [0-9]+ is out of range")
  bn=$(grep -cE "TT_FATAL.*cq_id [0-9]+ is out of range" "$d/runner.log" 2>/dev/null)
  if [ "$py" -ne 0 ] || [ "$tf" -ne 0 ]; then
    echo "  DIRTY $(basename $d): $py python-level, $tf non-benign TT_FATAL ($bn benign teardown)"
  else
    echo "  clean $(basename $d)  ($bn benign teardown TT_FATAL)"
  fi
done
