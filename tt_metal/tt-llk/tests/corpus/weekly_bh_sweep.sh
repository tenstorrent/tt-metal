#!/usr/bin/env bash
# Weekly BH 2x2 sweep entry point (cron-safe; see install_sweep_cron.sh).
#
# Nightly scope PLUS:
#  (a) per-knob attribution — every row whose .text changes OFF->ON is
#      re-classified with each optimization knob toggled individually
#      (latency-schedule, dst-iteration-fusion, replay-hoist, invariant-loadi,
#      dst-autoincr, macro-planner); headline rows (HEADLINE_ROWS in
#      sweep_2x2.conf) additionally get per-knob silicon legs;
#  (b) the WH CRAQ matrix for macro rows (craq_archs=bh,wh in
#      sweep_2x2_ops.tsv drives this through the same driver);
#  (c) the DejaGnu byte-parity suites (loadmacro*/macro-planner*) against the
#      pinned toolchain build tree when present, recorded as a machine-readable
#      SKIP when absent.
# Idempotent/resumable like the nightly; baselines are never modified.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=sweep_2x2.conf
source "$HERE/sweep_2x2.conf"

DATE=${SWEEP_DATE:-$(date +%Y%m%d)}
EV="$EVIDENCE_ROOT/weekly-$DATE"
BASELINE="$HERE/sfpu_device_baseline_${CHIP_CLASS}_v1.tsv"
[ -f "$BASELINE" ] || { echo "FATAL: no baseline for chip class '$CHIP_CLASS' ($BASELINE)"; exit 2; }
PREV=$(ls -d "$EVIDENCE_ROOT"/weekly-* 2>/dev/null | grep -vx "$EV" | sort | tail -1 || true)

echo "== weekly sweep $DATE -> $EV (prev: ${PREV:-none}) =="

python3 "$HERE/sfpu_corpus.py" --validate || { echo "FATAL: corpus validation failed"; exit 2; }

python3 "$HERE/sweep_2x2.py" \
  --evidence-root "$EV" \
  --compiler-sha "$PINNED_COMPILER_SHA256" \
  --sim-bh "$SIM_BH" --sim-wh "$SIM_WH" \
  --allow-hardware \
  --baseline "$BASELINE" \
  --max-drift-pct "$MAX_DRIFT_PCT" \
  --knob-attribution \
  --knob-silicon-rows "$HEADLINE_ROWS" \
  ${PREV:+--prev-run "$PREV"} \
  "$@"
RC=$?

# (c) DejaGnu byte-parity suites against the pinned toolchain build tree.
# Never clobber a build tree's evidentiary .sum: run from a scratch site dir.
DEJ="$EV/dejagnu"
mkdir -p "$DEJ"
if [ -x "$DEJAGNU_BUILD_TREE/gcc/xg++" ] && [ -d "$SFPI_GCC_SRC/gcc/testsuite" ] \
    && command -v runtest >/dev/null 2>&1; then
  if [ -s "$DEJ/summary.txt" ] && [ "${1:-}" != "--force" ]; then
    echo "dejagnu: resume — $DEJ/summary.txt exists"
  else
    cp "$DEJAGNU_BUILD_TREE/gcc/testsuite/g++/site.exp" "$DEJ/site.exp"
    echo "set GXX_UNDER_TEST \"$DEJAGNU_BUILD_TREE/gcc/xg++ -B$DEJAGNU_BUILD_TREE/gcc/\"" >> "$DEJ/site.exp"
    : > "$DEJ/summary.txt"
    for SUITE in $DEJAGNU_SUITES; do
      ( cd "$DEJ" && runtest --tool g++ --srcdir "$SFPI_GCC_SRC/gcc/testsuite" \
          "rvtt.exp=$SUITE" > "run-$SUITE.log" 2>&1 )
      PASS=$(grep -c '^PASS' "$DEJ/g++.sum" 2>/dev/null || echo 0)
      FAIL=$(grep -c '^FAIL' "$DEJ/g++.sum" 2>/dev/null || echo 0)
      cp "$DEJ/g++.sum" "$DEJ/g++-$SUITE.sum" 2>/dev/null
      echo -e "$SUITE\tPASS:$PASS\tFAIL:$FAIL" >> "$DEJ/summary.txt"
      # Byte-parity suites must be zero-FAIL (loadmacro*/macro-planner* carry
      # the minmax 19+6 parities and the refusal oracles).
      [ "$FAIL" -eq 0 ] || { echo "RED: dejagnu $SUITE has $FAIL FAILs"; RC=1; }
    done
  fi
else
  echo -e "dejagnu\tSKIP_NO_BUILD_TREE\t$DEJAGNU_BUILD_TREE" | tee "$DEJ/summary.txt"
fi

echo "== weekly sweep $DATE done rc=$RC; report: $EV/REPORT.md; dejagnu: $DEJ/summary.txt =="
exit $RC
