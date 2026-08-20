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

# --allow-pin-override: the ONLY sanctioned way to run with a PINNED_* value
# from the environment (sweep_2x2.conf rejects silent env overrides).  The
# flag is consumed here, never forwarded to sweep_2x2.py.
ARGS=()
for _a in "$@"; do
  if [ "$_a" = "--allow-pin-override" ]; then
    export ALLOW_PIN_OVERRIDE=1
    echo "weekly: --allow-pin-override — environment pin values will be honored AND LOGGED"
  else
    ARGS+=("$_a")
  fi
done
set -- ${ARGS[@]+"${ARGS[@]}"}

# conf-lint FIRST (enforcement layer, ledger item 10): the pin audit trail
# must agree before the conf is even sourced; the linter's own self-test
# runs first so a broken linter can never bless a sweep.
bash "$HERE/selftest_conf_lint.sh" > /tmp/weekly-selftest-conf-lint.$$ 2>&1 \
  || { echo "FATAL: conf-lint self-test failed:"; cat /tmp/weekly-selftest-conf-lint.$$; rm -f /tmp/weekly-selftest-conf-lint.$$; exit 2; }
bash "$HERE/conf_lint.sh" || { echo "FATAL: conf-lint refused — pin audit trail disagrees (fix conf prose/baseline header in the same commit as the pin change)"; exit 2; }

# shellcheck source=sweep_2x2.conf
source "$HERE/sweep_2x2.conf" || { echo "FATAL: sweep_2x2.conf refused (pin override without --allow-pin-override?)"; exit 2; }

DATE=${SWEEP_DATE:-$(date +%Y%m%d)}
EV="$EVIDENCE_ROOT/weekly-$DATE"
BASELINE="$HERE/sfpu_device_baseline_${CHIP_CLASS}_v1.tsv"
[ -f "$BASELINE" ] || { echo "FATAL: no baseline for chip class '$CHIP_CLASS' ($BASELINE)"; exit 2; }

# Evidence-root collision guard (incident 2026-08-20: the pin-14 weekly's
# date-derived root collided with the existing pin-12 weekly-20260820 — 15
# minutes of pin-14 classify wrote into pin-12 evidence).  The guard refuses
# any existing root recorded under a DIFFERENT toolchain pin and fails closed
# on unknown provenance; a same-pin root resumes as before.  SWEEP_DATE stays
# the manual root-name override (the refusal suggests a free one).
bash "$HERE/selftest_sweep_wrapper_lib.sh" > /tmp/weekly-selftest-wrapper-lib.$$ 2>&1 \
  || { echo "FATAL: sweep_wrapper_lib self-test failed:"; cat /tmp/weekly-selftest-wrapper-lib.$$; rm -f /tmp/weekly-selftest-wrapper-lib.$$; exit 2; }
# shellcheck source=sweep_wrapper_lib.sh
source "$HERE/sweep_wrapper_lib.sh" || { echo "FATAL: sweep_wrapper_lib.sh missing/broken"; exit 2; }
evidence_root_guard "$EV" "$PINNED_CC1PLUS_SHA256" "weekly_bh_sweep.sh" || exit 3
# --prev-run chain: the newest N clean run roots (weekly/nightly/headline,
# newest activity first; contaminated/quarantined roots are skipped).  Two
# consumers in sweep_2x2.py: the scoreboard annotator (newest root's
# scoreboard.json -> drift comparison) and the cross-pin cell-reuse prober
# (EVERY root probed for adoptable device cells; each source root is
# provenance-gated at adoption time — markers, pin record, craq-gate taint
# parity — and foreign pins adopt loudly with the pin recorded).
# SWEEP_PREV_CHAIN=N overrides the depth (default 3; 1 = old single-prev
# behavior + clean filter).
PREV=$(newest_clean_runs "$EVIDENCE_ROOT" "$EV" "${SWEEP_PREV_CHAIN:-3}" weekly nightly headline)

echo "== weekly sweep $DATE -> $EV (prev: ${PREV:-none}) =="

python3 "$HERE/sfpu_corpus.py" --validate || { echo "FATAL: corpus validation failed"; exit 2; }

# Gate self-tests first: a broken gate must never bless a sweep.
mkdir -p "$EV"
python3 "$HERE/selftest_sweep_2x2_report.py" > "$EV/selftest-report-gate.txt" 2>&1 \
  || { echo "FATAL: report-gate self-test failed (see $EV/selftest-report-gate.txt)"; exit 2; }
python3 "$HERE/selftest_enforcement_gates.py" > "$EV/selftest-enforcement-gates.txt" 2>&1 \
  || { echo "FATAL: enforcement-gates self-test failed (see $EV/selftest-enforcement-gates.txt)"; exit 2; }
python3 "$HERE/selftest_sweep_core_overhaul.py" > "$EV/selftest-sweep-core-overhaul.txt" 2>&1 \
  || { echo "FATAL: sweep-core-overhaul self-test failed (see $EV/selftest-sweep-core-overhaul.txt)"; exit 2; }
{ mv /tmp/weekly-selftest-conf-lint.$$ "$EV/selftest-conf-lint.txt" 2>/dev/null || true; }
{ mv /tmp/weekly-selftest-wrapper-lib.$$ "$EV/selftest-wrapper-lib.txt" 2>/dev/null || true; }
bash "$HERE/conf_lint.sh" > "$EV/conf-lint.txt" 2>&1 \
  || { echo "FATAL: conf-lint refused (see $EV/conf-lint.txt)"; exit 2; }

# No --schedule filter here: the weekly sweep deliberately runs EVERY ops.tsv
# row, including the schedule=weekly deferrals the nightly skips (the
# device-time budget split is data in the TSV's schedule column, not a fork).
# RATIFIED (owner, 2026-08-20; charter §1(3) amended same day): sweeps run
# STRAIGHT SILICON — per-cell device-golden correctness legs gate every perf
# cell; CRAQ is a debug/lane-validation oracle (pinned sims), not a sweep
# precondition.  This flag is therefore the sanctioned default here, and
# sweep_2x2.py records an explicit CRAQ-gate taint/status line either way.
# Unbuffered: without this the sweep's stdout block-buffers into the tee'd
# log (a setsid-detached sweep looks dead for many minutes between flushes).
# PYTHONUNBUFFERED covers python and every python child (pytest sessions);
# stdbuf -oL -eL covers any C-stdio subprocess in between.
PYTHONUNBUFFERED=1 stdbuf -oL -eL python3 "$HERE/sweep_2x2.py" \
  --evidence-root "$EV" \
  --cc1plus-sha "$PINNED_CC1PLUS_SHA256" \
  --compiler-sha "$PINNED_COMPILER_SHA256" \
  --sim-bh "$SIM_BH" --sim-wh "$SIM_WH" \
  --sim-bh-sha "$PINNED_SIM_BH_SHA256" --sim-wh-sha "$PINNED_SIM_WH_SHA256" \
  --phases "${SWEEP_PHASES:-classify,silicon,report}" \
  --skip-craq-gate \
  --allow-hardware \
  --baseline "$BASELINE" \
  --max-drift-pct "$MAX_DRIFT_PCT" \
  --max-abs-drift-pct "$MAX_ABS_DRIFT_PCT" \
  --red-loss-growth-pct "$RED_LOSS_GROWTH_PCT" \
  --knob-attribution \
  --knob-silicon-rows "$HEADLINE_ROWS" \
  ${PREV:+--prev-run "$PREV"} \
  "$@"
RC=$?

# (c) DejaGnu byte-parity suites against the pinned toolchain build tree.
# The counting/gating logic lives in dejagnu_gate.sh (self-tested by
# selftest_dejagnu_gate.sh: clean->GREEN, failing->RED, no-sum->RED); the
# self-test runs first so a broken gate can never bless tonight's suites.
mkdir -p "$EV"
bash "$HERE/selftest_dejagnu_gate.sh" > "$EV/selftest-dejagnu-gate.txt" 2>&1 \
  || { echo "RED: dejagnu gate self-test failed (see $EV/selftest-dejagnu-gate.txt)"; RC=1; }
DEJAGNU_BUILD_TREE="$DEJAGNU_BUILD_TREE" SFPI_GCC_SRC="$SFPI_GCC_SRC" \
  DEJAGNU_SUITES="$DEJAGNU_SUITES" bash "$HERE/dejagnu_gate.sh" "$EV" "${1:-}" || RC=1

echo "== weekly sweep $DATE done rc=$RC; report: $EV/REPORT.md; dejagnu: $EV/dejagnu/summary.txt =="
exit $RC
