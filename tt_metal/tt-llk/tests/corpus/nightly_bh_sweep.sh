#!/usr/bin/env bash
# Nightly BH 2x2 sweep entry point (cron-safe; see install_sweep_cron.sh).
#
# Scope: pinned-toolchain verification, corpus inventory validation,
# compile-mode OFF/ON classification of every mapped sweep row, paired CRAQ on
# changed pairs, then the serialized BH silicon 2x2 for the mapped perf rows
# (both flocks per HANDOFF §1(5)), a dated evidence dir with the same manifest
# discipline as the one-command sweep, and a REPORT.md with three comparisons:
# checked-in chip-class baseline, previous nightly run, and the §1 acceptance
# verdicts (win-sign preserved / refusal byte-identical / flip = RED).
#
# Assumes the pinned toolchain and simulator are already built; verifies their
# identity and aborts loudly on mismatch (sweep_2x2.py preflight).
# Idempotent and resumable per row/job: re-running the same SWEEP_DATE skips
# evidence that already exists unless --force is passed through.
# Exit nonzero on any RED so a cron wrapper can alert.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

# --allow-pin-override: the ONLY sanctioned way to run with a PINNED_* value
# from the environment (sweep_2x2.conf rejects silent env overrides).  The
# flag is consumed here, never forwarded to sweep_2x2.py.
# --skip-witness: the ONLY sanctioned way to skip the union fire-witness
# preflight (conf R9's compile half, witness_preflight.py) — an EMERGENCY
# escape, logged loudly here and recorded in the evidence dir.
ARGS=()
SKIP_WITNESS=0
for _a in "$@"; do
  if [ "$_a" = "--allow-pin-override" ]; then
    export ALLOW_PIN_OVERRIDE=1
    echo "nightly: --allow-pin-override — environment pin values will be honored AND LOGGED"
  elif [ "$_a" = "--skip-witness" ]; then
    SKIP_WITNESS=1
    echo "nightly: *** --skip-witness — THE UNION FIRE-WITNESS GATE IS BEING SKIPPED (emergency escape; the pin-11 no-fire class is UNGUARDED this run) ***"
  else
    ARGS+=("$_a")
  fi
done
set -- ${ARGS[@]+"${ARGS[@]}"}

# conf-lint FIRST (enforcement layer, ledger item 10): the pin audit trail
# (conf values ↔ prose ↔ PIN HISTORY ↔ baseline header) must agree before the
# conf is even sourced — a lying audit trail refuses the whole sweep.  The
# linter's own self-test runs first so a broken linter can never bless one.
bash "$HERE/selftest_conf_lint.sh" > /tmp/nightly-selftest-conf-lint.$$ 2>&1 \
  || { echo "FATAL: conf-lint self-test failed:"; cat /tmp/nightly-selftest-conf-lint.$$; rm -f /tmp/nightly-selftest-conf-lint.$$; exit 2; }
bash "$HERE/conf_lint.sh" || { echo "FATAL: conf-lint refused — pin audit trail disagrees (fix conf prose/baseline header in the same commit as the pin change)"; exit 2; }

# shellcheck source=sweep_2x2.conf
source "$HERE/sweep_2x2.conf" || { echo "FATAL: sweep_2x2.conf refused (pin override without --allow-pin-override?)"; exit 2; }

DATE=${SWEEP_DATE:-$(date +%Y%m%d)}
EV="$EVIDENCE_ROOT/nightly-$DATE"
BASELINE="$HERE/sfpu_device_baseline_${CHIP_CLASS}_v1.tsv"
[ -f "$BASELINE" ] || { echo "FATAL: no baseline for chip class '$CHIP_CLASS' ($BASELINE)"; exit 2; }
# KERNEL-scoped (v2) VERDICT baseline (lane ET, owner ratification
# 2026-08-21): passed when seeded; absent = bootstrap (kernel ratios report
# no-baseline, v1 diagnostic checks keep full severity — handover rule).
KBASELINE="$HERE/sfpu_device_baseline_${CHIP_CLASS}_v2.tsv"
[ -f "$KBASELINE" ] || KBASELINE=""

# Evidence-root collision guard (incident 2026-08-20: the pin-14 weekly's
# date-derived root collided with the existing pin-12 weekly-20260820 — 15
# minutes of pin-14 classify wrote into pin-12 evidence).  The guard refuses
# any existing root recorded under a DIFFERENT toolchain pin and fails closed
# on unknown provenance; a same-pin root resumes as before.  SWEEP_DATE stays
# the manual root-name override (the refusal suggests a free one).
bash "$HERE/selftest_sweep_wrapper_lib.sh" > /tmp/nightly-selftest-wrapper-lib.$$ 2>&1 \
  || { echo "FATAL: sweep_wrapper_lib self-test failed:"; cat /tmp/nightly-selftest-wrapper-lib.$$; rm -f /tmp/nightly-selftest-wrapper-lib.$$; exit 2; }
# shellcheck source=sweep_wrapper_lib.sh
source "$HERE/sweep_wrapper_lib.sh" || { echo "FATAL: sweep_wrapper_lib.sh missing/broken"; exit 2; }
evidence_root_guard "$EV" "$PINNED_CC1PLUS_SHA256" "nightly_bh_sweep.sh" || exit 3

# --prev-run chain: newest N clean run roots across nightly/weekly/headline
# (contaminated/quarantined skipped).  Consumed twice by sweep_2x2.py: the
# scoreboard annotator (newest root's scoreboard.json -> drift comparison)
# and the cross-pin cell-reuse prober (EVERY root probed for adoptable
# device cells; source roots are provenance-gated at adoption time —
# markers, pin record, craq-gate taint parity; foreign pins adopt loudly
# with the pin recorded).  SWEEP_PREV_CHAIN=N overrides the depth (default 3).
PREV=$(newest_clean_runs "$EVIDENCE_ROOT" "$EV" "${SWEEP_PREV_CHAIN:-3}" nightly weekly headline)

echo "== nightly sweep $DATE -> $EV (prev: ${PREV:-none}) =="

# Corpus inventory must be self-consistent before any measurement.
python3 "$HERE/sfpu_corpus.py" --validate || { echo "FATAL: corpus validation failed"; exit 2; }

# Gate self-tests FIRST: the class-aware flip detector (win->refusal must be
# RED — selftest_sweep_2x2_report.py drives the real report()) and the
# DejaGnu counting gate (clean->GREEN/failing->RED — weekly's suites, but the
# shared logic is proven nightly too).  A broken gate must never bless a
# sweep; the outputs are appended to REPORT.md below.
mkdir -p "$EV"
GATE_SELFTEST_RC=0
python3 "$HERE/selftest_sweep_2x2_report.py" > "$EV/selftest-report-gate.txt" 2>&1 \
  || GATE_SELFTEST_RC=1
bash "$HERE/selftest_dejagnu_gate.sh" > "$EV/selftest-dejagnu-gate.txt" 2>&1 \
  || GATE_SELFTEST_RC=1
python3 "$HERE/selftest_enforcement_gates.py" > "$EV/selftest-enforcement-gates.txt" 2>&1 \
  || GATE_SELFTEST_RC=1
python3 "$HERE/selftest_witness_preflight.py" > "$EV/selftest-witness-preflight.txt" 2>&1 \
  || GATE_SELFTEST_RC=1
python3 "$HERE/selftest_batched_silicon.py" > "$EV/selftest-batched-silicon.txt" 2>&1 \
  || GATE_SELFTEST_RC=1
python3 "$HERE/selftest_sweep_core_overhaul.py" > "$EV/selftest-sweep-core-overhaul.txt" 2>&1 \
  || GATE_SELFTEST_RC=1
python3 "$HERE/selftest_knob_legs_semleg.py" > "$EV/selftest-knob-legs-semleg.txt" 2>&1 \
  || GATE_SELFTEST_RC=1
python3 "$HERE/selftest_dst_layout_32b.py" > "$EV/selftest-dst-layout-32b.txt" 2>&1 \
  || GATE_SELFTEST_RC=1
python3 "$HERE/selftest_e2e_metric.py" > "$EV/selftest-e2e-metric.txt" 2>&1 \
  || { echo "FATAL: e2e-metric (dual-zone verdict) self-test failed (see $EV/selftest-e2e-metric.txt)"; exit 2; }
python3 "$HERE/selftest_perf_schema_columns.py" > "$EV/selftest-perf-schema-columns.txt" 2>&1 \
  || { echo "FATAL: perf-schema-columns self-test failed (see $EV/selftest-perf-schema-columns.txt)"; exit 2; }
# Upstream perf header gate (FO-1): schema catalog + global field uniqueness +
# duplicate-param-type checks. FATAL so it can never drift silently again; a
# missing tests venv is FATAL too (fail-closed).
HDRGATE_PY="$HERE/../python_tests/.venv/bin/python"
[ -x "$HDRGATE_PY" ] \
  || { echo "FATAL: perf header gate needs the tests venv ($HDRGATE_PY missing)"; exit 2; }
( cd "$HERE/../python_tests" && "$HDRGATE_PY" -m pytest -q test_perf_header_gate.py ) \
  > "$EV/selftest-perf-header-gate.txt" 2>&1 \
  || { echo "FATAL: perf header gate RED (see $EV/selftest-perf-header-gate.txt)"; exit 2; }
# Record the conf-lint verdict (already enforced above, pre-source) in-evidence.
{ mv /tmp/nightly-selftest-conf-lint.$$ "$EV/selftest-conf-lint.txt" 2>/dev/null || true; }
{ mv /tmp/nightly-selftest-wrapper-lib.$$ "$EV/selftest-wrapper-lib.txt" 2>/dev/null || true; }
bash "$HERE/conf_lint.sh" > "$EV/conf-lint.txt" 2>&1 || GATE_SELFTEST_RC=1
if [ "$GATE_SELFTEST_RC" -ne 0 ]; then
  echo "FATAL: gate self-tests failed (see $EV/selftest-*.txt) — refusing to sweep"
  exit 2
fi

# UNION FIRE-WITNESS preflight (conf R9's compile half; the pin-11 lesson):
# every _REVIEWED_FIRE_WITNESSES entry's node is compiled at the pinned
# toolchain with the FULL reviewed ON set + its dump flag, and the required
# dump line must be present — a missing line is RED naming the flag ('fire
# witness stale on the union') and refuses the sweep.  Fast (~1-3 compiles,
# witness nodes only).  SKIP-with-reason when the env preconditions are
# absent (same policy as the corpus compile gate below) or on --dry-run;
# --skip-witness is the loudly-logged emergency escape.
WIT_STATUS="" WIT_REASON=""
WIT_PY="$HERE/../python_tests/.venv/bin/python"
WIT_CXX="$HERE/../sfpi/compiler/bin/riscv-tt-elf-g++"
WIT_DRY=0
for _a in "$@"; do [ "$_a" = "--dry-run" ] && WIT_DRY=1; done
if [ "$SKIP_WITNESS" = 1 ]; then
  WIT_STATUS=SKIPPED WIT_REASON="--skip-witness EMERGENCY ESCAPE (the pin-11 no-fire class is unguarded this run)"
elif [ ! -x "$WIT_PY" ]; then
  WIT_STATUS=SKIP WIT_REASON="missing tt-llk venv ($WIT_PY)"
elif [ ! -x "$WIT_CXX" ]; then
  WIT_STATUS=SKIP WIT_REASON="missing pinned SFPI toolchain ($WIT_CXX)"
elif [ "$WIT_DRY" = 1 ]; then
  WIT_STATUS=DRY_RUN WIT_REASON="dry-run: witness compiles not executed; real command: python3 $HERE/witness_preflight.py --work $EV/witness-preflight"
else
  if python3 "$HERE/witness_preflight.py" --work "$EV/witness-preflight" \
       > "$EV/witness-preflight.txt" 2>&1; then
    WIT_STATUS=PASS WIT_REASON="every declared witness fires on the union ($EV/witness-preflight/verdicts.json)"
  else
    WIT_RC=$?
    echo "RED: union fire-witness preflight FAILED (rc=$WIT_RC) — an ON-set flag's fire witness is stale on the union (or the gate could not run):"
    grep -E "RED|ERROR" "$EV/witness-preflight.txt" | head -10 || true
    echo "     (full output: $EV/witness-preflight.txt; --skip-witness is the logged emergency escape)"
    echo "witness-preflight: RED (rc=$WIT_RC)" > "$EV/witness-preflight-status.txt"
    exit 1
  fi
fi
echo "witness-preflight: $WIT_STATUS — $WIT_REASON" | tee "$EV/witness-preflight-status.txt"

# Corpus compile gate (coverage-parity plan item 1): every mapped corpus row
# (109 functional-module-mapped rows) must COMPILE green on BH with the pinned
# toolchain before any 2x2 phase runs — `--require-executed-mapped` turns a
# single non-PASS mapped row into a nonzero exit (RED).  The gate is
# compile-mode only (no simulator, no device), so it is safe to run before the
# flocked phases.  CRAQ IS DEBUG-ONLY (owner ruling 2026-08-19): the
# measurement path gates correctness on the DEVICE GOLDEN legs; set
# SWEEP_PHASES=classify,craq,silicon,report to re-enable CRAQ for
# debugging a sim-vs-silicon divergence.  SKIP-with-reason when the env preconditions are absent
# (recorded in the report, never silent); a completed PASS gate for the same
# SWEEP_DATE is reused (idempotent resume, matching the per-row discipline).
# A --dry-run wrapper invocation prints the real gate command and proves the
# wiring with a plan-only pass instead of the full compile.
GATE_STATUS="" GATE_REASON=""
GATE_ROOT="$EV/corpus-compile-gate"
GATE_DRY=0
for _a in "$@"; do [ "$_a" = "--dry-run" ] && GATE_DRY=1; done
GATE_PY="$HERE/../python_tests/.venv/bin/python"
GATE_CXX="$HERE/../sfpi/compiler/bin/riscv-tt-elf-g++"
GATE_CMD=(python3 "$HERE/sfpu_corpus.py" --mode compile --arch bh --execute --require-executed-mapped)
if [ ! -x "$GATE_PY" ]; then
  GATE_STATUS=SKIP GATE_REASON="missing tt-llk venv ($GATE_PY)"
elif [ ! -x "$GATE_CXX" ]; then
  GATE_STATUS=SKIP GATE_REASON="missing pinned SFPI toolchain ($GATE_CXX)"
elif [ "$GATE_DRY" = 1 ]; then
  echo "nightly: corpus compile gate DRY-RUN — real command would be:"
  echo "  ${GATE_CMD[*]} --run-root $GATE_ROOT"
  rm -rf "$GATE_ROOT.dry"
  if python3 "$HERE/sfpu_corpus.py" --mode compile --arch bh \
       --run-root "$GATE_ROOT.dry" > "$EV/corpus-compile-gate-dry.log" 2>&1; then
    GATE_STATUS=DRY_RUN GATE_REASON="plan-only wiring proof (no --execute); see $GATE_ROOT.dry"
  else
    echo "RED: corpus compile gate dry-run (plan-only) failed (see $EV/corpus-compile-gate-dry.log)"
    exit 1
  fi
elif [ -f "$GATE_ROOT/results.json" ] && \
     python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["provenance"].get("executed_mapped_gate")=="PASS" else 1)' \
       "$GATE_ROOT/results.json" 2>/dev/null; then
  GATE_STATUS=PASS GATE_REASON="reused: executed_mapped_gate already PASS for $DATE"
else
  # sfpu_corpus.py refuses a pre-existing --run-root; rotate a stale/failed one.
  [ -e "$GATE_ROOT" ] && mv "$GATE_ROOT" "$GATE_ROOT.retry-$(date +%H%M%S)"
  if "${GATE_CMD[@]}" --run-root "$GATE_ROOT" > "$EV/corpus-compile-gate.log" 2>&1; then
    GATE_STATUS=PASS GATE_REASON="all mapped rows compiled PASS ($GATE_ROOT/results.tsv)"
  else
    echo "RED: corpus compile gate FAILED — a mapped corpus row did not compile PASS"
    echo "     (see $GATE_ROOT/results.tsv and $EV/corpus-compile-gate.log)"
    echo "corpus-compile-gate: RED" > "$EV/corpus-compile-gate-status.txt"
    exit 1
  fi
fi
echo "corpus-compile-gate: $GATE_STATUS — $GATE_REASON" | tee "$EV/corpus-compile-gate-status.txt"

# sweep_2x2.py preflight enforces: cc1plus sha256 == PRIMARY pin (resolved
# via g++ -print-prog-name=cc1plus; the driver sha is a secondary check —
# the driver is byte-identical across cc1plus-only changes), removed
# loadmacro flags error on use, OFF/ON flag sets accepted. It classifies
# BEFORE any device job, refuses byte-identical pairs, gates every perf cell on
# per-cell device-golden correctness legs (straight silicon, ratified; CRAQ is a
# pinned-sim debug/lane oracle, skip-runs taint-marked), serializes every device job under both flocks, copies CSVs
# in-lock, and never parses results while a lock is held. Resume is
# hash-matched: cached cells are reused only when their archived .text
# hashes equal this run's classify build. Baselines are read-only here.
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
  ${KBASELINE:+--kernel-baseline "$KBASELINE"} \
  --max-drift-pct "$MAX_DRIFT_PCT" \
  --max-abs-drift-pct "$MAX_ABS_DRIFT_PCT" \
  --red-loss-growth-pct "$RED_LOSS_GROWTH_PCT" \
  --schedule nightly \
  ${PREV:+--prev-run "$PREV"} \
  "$@"
RC=$?

# Append the gate self-test evidence to the report so every nightly verdict
# carries the proof that its flip detector works.
if [ -f "$EV/REPORT.md" ]; then
  {
    echo ""
    echo "## Gate self-tests (run before the sweep)"
    echo ""
    echo '```'
    tail -n 3 "$EV/selftest-report-gate.txt"
    tail -n 1 "$EV/selftest-dejagnu-gate.txt"
    tail -n 1 "$EV/selftest-enforcement-gates.txt" 2>/dev/null || echo "enforcement-gates self-test: (no record)"
    tail -n 1 "$EV/selftest-witness-preflight.txt" 2>/dev/null || echo "witness-preflight self-test: (no record)"
    tail -n 1 "$EV/selftest-batched-silicon.txt" 2>/dev/null || echo "batched-silicon self-test: (no record)"
    tail -n 1 "$EV/selftest-conf-lint.txt" 2>/dev/null || true
    tail -n 1 "$EV/conf-lint.txt" 2>/dev/null || echo "conf-lint: (no record)"
    cat "$EV/witness-preflight-status.txt" 2>/dev/null || echo "witness-preflight: (no status recorded)"
    cat "$EV/corpus-compile-gate-status.txt" 2>/dev/null || echo "corpus-compile-gate: (no status recorded)"
    echo '```'
  } >> "$EV/REPORT.md"
fi

echo "== nightly sweep $DATE done rc=$RC; report: $EV/REPORT.md =="
exit $RC
