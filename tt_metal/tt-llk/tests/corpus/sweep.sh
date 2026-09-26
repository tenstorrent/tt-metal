#!/usr/bin/env bash
# sweep.sh — THE corpus sweep entry point.  One command, three modes.
#
#   bash sweep.sh --mode headline   # flip-prone rows only          (minutes)
#   bash sweep.sh --mode nightly    # every mapped row              (~1 h)
#   bash sweep.sh --mode weekly     # nightly + knob attribution
#                                   #   + DejaGnu byte-parity       (~3 h)
#
# Replaces headline_bh_sweep.sh / nightly_bh_sweep.sh / weekly_bh_sweep.sh,
# which were 616 lines ending in a byte-identical sweep_2x2.py invocation
# with a one-or-two-flag delta:
#
#   headline  --ops <derived> [--priority-ops <headline∩ops>]
#   nightly   --schedule nightly
#   weekly    --knob-attribution --knob-silicon-rows "$HEADLINE_ROWS"
#
# Everything else that differed was preset (evidence-root prefix, prev-run
# preference order, which gates run, gate-failure policy) and is now a
# `case "$MODE"`.  sweep_2x2.py owns row selection, schedule filtering,
# priority ordering, all four phases, every toolchain/sim/review-record
# gate, resume-by-hash, cross-run cell reuse and all report artifacts; this
# wrapper supplies only what the engine cannot: conf sourcing, the three
# shell preflights (conf-lint, witness, DejaGnu), the evidence-root
# provenance guard, run naming, and the mode preset.
#
# Any argument that is not a wrapper flag is forwarded verbatim to
# sweep_2x2.py, so `--ops a,b --phases classify --dry-run --force` all work
# unchanged.
#
# Exit codes: 0 green · 1 sweep RED / witness RED / compile-gate RED /
#             DejaGnu RED · 2 a gate or self-test refused · 3 evidence-root
#             collision.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

MODE=""
SKIP_WITNESS=0
ARGS=()

usage() {
  sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-2}"
}

# ---------------------------------------------------------------- flags --
# Consumed here, never forwarded:
#   --mode M              which preset to run (required)
#   --allow-pin-override  the ONLY sanctioned way to run with a PINNED_*
#                         value from the environment (sweep_2x2.conf rejects
#                         silent env overrides)
#   --skip-witness        EMERGENCY escape from the union fire-witness
#                         preflight (nightly only); logged loudly and
#                         recorded in the evidence dir
while [ $# -gt 0 ]; do
  case "$1" in
    --mode)       MODE=${2:-}; shift 2 || usage 2 ;;
    --mode=*)     MODE=${1#--mode=}; shift ;;
    -h|--help)    usage 0 ;;
    --allow-pin-override)
      export ALLOW_PIN_OVERRIDE=1
      echo "sweep: --allow-pin-override — environment pin values will be honored AND LOGGED"
      shift ;;
    --skip-witness)
      SKIP_WITNESS=1
      echo "sweep: *** --skip-witness — THE UNION FIRE-WITNESS GATE IS BEING SKIPPED (emergency escape; the pin-11 no-fire class is UNGUARDED this run) ***"
      shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
set -- ${ARGS[@]+"${ARGS[@]}"}

case "$MODE" in
  headline|nightly|weekly) ;;
  "") echo "FATAL: --mode is required (headline|nightly|weekly)" >&2; usage 2 ;;
  *)  echo "FATAL: unknown --mode '$MODE' (headline|nightly|weekly)" >&2; exit 2 ;;
esac

DRY_RUN=0
for _a in "$@"; do [ "$_a" = "--dry-run" ] && DRY_RUN=1; done

# ------------------------------------------------------- conf bootstrap --
# conf-lint FIRST (enforcement layer, ledger item 10): the pin audit trail
# (conf values <-> prose <-> PIN HISTORY <-> baseline header) must agree
# before the conf is even sourced.  The linter's own self-test runs first —
# a broken linter can never bless a sweep.
_CONF_LINT_TMP=/tmp/sweep-$MODE-selftest-conf-lint.$$
bash "$HERE/selftest_conf_lint.sh" > "$_CONF_LINT_TMP" 2>&1 \
  || { echo "FATAL: conf-lint self-test failed:"; cat "$_CONF_LINT_TMP"; rm -f "$_CONF_LINT_TMP"; exit 2; }
bash "$HERE/conf_lint.sh" \
  || { echo "FATAL: conf-lint refused — pin audit trail disagrees (fix conf prose/baseline header in the same commit as the pin change)"; exit 2; }

# shellcheck source=sweep_2x2.conf
source "$HERE/sweep_2x2.conf" \
  || { echo "FATAL: sweep_2x2.conf refused (pin override without --allow-pin-override?)"; exit 2; }

# --------------------------------------------------------- run identity --
DATE=${SWEEP_DATE:-$(date +%Y%m%d)}
EV="$EVIDENCE_ROOT/$MODE-$DATE"
BASELINE="$HERE/sfpu_device_baseline_${CHIP_CLASS}_v1.tsv"
[ -f "$BASELINE" ] || { echo "FATAL: no baseline for chip class '$CHIP_CLASS' ($BASELINE)"; exit 2; }
# KERNEL-scoped (v2) VERDICT baseline: passed when seeded; absent = bootstrap
# (kernel ratios report no-baseline, v1 diagnostic checks keep full severity).
KBASELINE="$HERE/sfpu_device_baseline_${CHIP_CLASS}_v2.tsv"
[ -f "$KBASELINE" ] || KBASELINE=""

# Wrapper-lib self-test, then the evidence-root collision guard (incident
# 2026-08-20: pin-14 classify wrote 15 min into the pin-12 weekly-20260820
# root).  SWEEP_DATE stays the manual root-name override.
_LIB_TMP=/tmp/sweep-$MODE-selftest-wrapper-lib.$$
bash "$HERE/selftest_sweep_wrapper_lib.sh" > "$_LIB_TMP" 2>&1 \
  || { echo "FATAL: sweep_wrapper_lib self-test failed:"; cat "$_LIB_TMP"; rm -f "$_LIB_TMP"; exit 2; }
# shellcheck source=sweep_wrapper_lib.sh
source "$HERE/sweep_wrapper_lib.sh" || { echo "FATAL: sweep_wrapper_lib.sh missing/broken"; exit 2; }
evidence_root_guard "$EV" "$PINNED_CC1PLUS_SHA256" "sweep.sh --mode $MODE" || exit 3

# --prev-run chain: newest N clean roots, own mode first so a resume prefers
# its own lineage.  Consumed twice by sweep_2x2.py — scoreboard drift
# annotation (newest root) and cross-pin cell reuse (every root probed;
# sources are provenance-gated at adoption time).
case "$MODE" in
  headline) PREV_ORDER=(headline weekly nightly) ;;
  nightly)  PREV_ORDER=(nightly weekly headline) ;;
  weekly)   PREV_ORDER=(weekly nightly headline) ;;
esac
PREV=$(newest_clean_runs "$EVIDENCE_ROOT" "$EV" "${SWEEP_PREV_CHAIN:-3}" "${PREV_ORDER[@]}")

echo "== $MODE sweep $DATE -> $EV (prev chain: ${PREV:-none}) =="

python3 "$HERE/sfpu_corpus.py" --validate || { echo "FATAL: corpus validation failed"; exit 2; }

# ------------------------------------------------------ gate self-tests --
# A broken gate must never bless a run.  headline/weekly fail fast; nightly
# accumulates so one cron run reports every broken gate at once — except the
# three that stay fail-fast in every mode (e2e-metric, perf-schema-columns,
# perf-header-gate), which guard the verdict arithmetic itself.
mkdir -p "$EV"
mv "$_LIB_TMP" "$EV/selftest-wrapper-lib.txt" 2>/dev/null || true
mv "$_CONF_LINT_TMP" "$EV/selftest-conf-lint.txt" 2>/dev/null || true

GATE_RC=0
# selftest <runner> <script> <evidence-name> <description>
selftest() {
  local runner=$1 script=$2 name=$3 desc=$4
  "$runner" "$HERE/$script" > "$EV/$name.txt" 2>&1 && return 0
  if [ "$MODE" = nightly ]; then
    GATE_RC=1
    echo "RED: $desc self-test failed (see $EV/$name.txt)"
  else
    echo "FATAL: $desc self-test failed (see $EV/$name.txt)"; exit 2
  fi
}
# fatal_selftest — fail-fast in every mode.
fatal_selftest() {
  local runner=$1 script=$2 name=$3 desc=$4
  "$runner" "$HERE/$script" > "$EV/$name.txt" 2>&1 \
    || { echo "FATAL: $desc self-test failed (see $EV/$name.txt)"; exit 2; }
}

selftest python3 selftest_sweep_2x2_report.py    selftest-report-gate        "report-gate"
selftest python3 selftest_enforcement_gates.py   selftest-enforcement-gates  "enforcement-gates"
if [ "$MODE" = nightly ]; then
  selftest bash    selftest_dejagnu_gate.sh      selftest-dejagnu-gate       "dejagnu-gate"
  selftest python3 selftest_witness_preflight.py selftest-witness-preflight  "witness-preflight"
  selftest python3 selftest_batched_silicon.py   selftest-batched-silicon    "batched-silicon"
fi
if [ "$MODE" = nightly ] || [ "$MODE" = weekly ]; then
  selftest python3 selftest_sweep_core_overhaul.py selftest-sweep-core-overhaul "sweep-core-overhaul"
fi
selftest python3 selftest_knob_legs_semleg.py    selftest-knob-legs-semleg   "knob-legs/sem-leg"
selftest python3 selftest_dst_layout_32b.py      selftest-dst-layout-32b     "dst-layout-32b wiring"
fatal_selftest python3 selftest_e2e_metric.py          selftest-e2e-metric          "e2e-metric (dual-zone verdict)"
fatal_selftest python3 selftest_perf_schema_columns.py selftest-perf-schema-columns "perf-schema-columns"

# Upstream perf header gate (FO-1): schema catalog + global field uniqueness
# + duplicate-param-type checks.  A missing tests venv is FATAL too
# (fail-closed) — this gate must never drift silently again.
HDRGATE_PY="$HERE/../python_tests/.venv/bin/python"
[ -x "$HDRGATE_PY" ] \
  || { echo "FATAL: perf header gate needs the tests venv ($HDRGATE_PY missing)"; exit 2; }
( cd "$HERE/../python_tests" && "$HDRGATE_PY" -m pytest -q test_perf_header_gate.py ) \
  > "$EV/selftest-perf-header-gate.txt" 2>&1 \
  || { echo "FATAL: perf header gate RED (see $EV/selftest-perf-header-gate.txt)"; exit 2; }

bash "$HERE/conf_lint.sh" > "$EV/conf-lint.txt" 2>&1 || {
  if [ "$MODE" = nightly ]; then GATE_RC=1
  else echo "FATAL: conf-lint refused (see $EV/conf-lint.txt)"; exit 2; fi
}
if [ "$GATE_RC" -ne 0 ]; then
  echo "FATAL: gate self-tests failed (see $EV/selftest-*.txt) — refusing to sweep"
  exit 2
fi

# ------------------------------------------------------- mode pre-gates --
DELTA=()

if [ "$MODE" = headline ]; then
  # Ops: explicit --ops wins; otherwise derive headline ∪ changed-since-last-
  # pin rows from git (derivation log kept in evidence).
  HAVE_OPS=0 EXPLICIT_OPS="" _prev=""
  for _a in "$@"; do
    [ "$_prev" = "--ops" ] && EXPLICIT_OPS=$_a
    [ "$_a" = "--ops" ] && HAVE_OPS=1
    case "$_a" in --ops=*) HAVE_OPS=1; EXPLICIT_OPS=${_a#--ops=};; esac
    _prev=$_a
  done
  if [ "$HAVE_OPS" = 1 ]; then
    echo "headline: explicit --ops passed — skipping git derivation"
  else
    OPS=$(python3 "$HERE/headline_ops.py" --headline "$HEADLINE_ROWS" \
          2> "$EV/headline-ops-derivation.txt") \
      || { echo "FATAL: headline_ops.py failed:"; cat "$EV/headline-ops-derivation.txt"; exit 2; }
    echo "headline: ops = $OPS"
    echo "headline: derivation log: $EV/headline-ops-derivation.txt"
    DELTA+=(--ops "$OPS")
  fi

  # --priority-ops: measure the flip-prone headline rows first.  With an
  # explicit --ops list, pass only the headline rows INSIDE it — sweep_2x2.py
  # refuses a --priority-ops row outside --ops ("unknown ops", rc=1) AFTER
  # the evidence root is stamped, which contaminates the root.
  PRIO_ROWS=$HEADLINE_ROWS
  if [ "$HAVE_OPS" = 1 ]; then
    PRIO_ROWS=$(python3 -c '
import sys
head = [o for o in sys.argv[1].split(",") if o]
ops = set(o for o in sys.argv[2].split(",") if o)
print(",".join(o for o in head if o in ops))
' "$HEADLINE_ROWS" "$EXPLICIT_OPS")
  fi
  if [ -n "$PRIO_ROWS" ]; then
    DELTA+=(--priority-ops "$PRIO_ROWS")
    echo "headline: priority rows: $PRIO_ROWS"
  else
    echo "headline: explicit --ops shares no rows with HEADLINE_ROWS — no --priority-ops"
  fi
fi

if [ "$MODE" = nightly ]; then
  DELTA+=(--schedule nightly)

  # UNION FIRE-WITNESS preflight (conf R9's compile half; the pin-11 lesson):
  # every _REVIEWED_FIRE_WITNESSES entry's node is compiled at the pinned
  # toolchain with the FULL reviewed ON set + its dump flag, and the required
  # dump line must be present — a missing line is RED naming the flag and
  # refuses the sweep.  SKIP-with-reason when the env preconditions are
  # absent; --skip-witness is the loudly-logged emergency escape.
  WIT_STATUS="" WIT_REASON=""
  WIT_PY="$HERE/../python_tests/.venv/bin/python"
  WIT_CXX="$HERE/../sfpi/compiler/bin/riscv-tt-elf-g++"
  if [ "$SKIP_WITNESS" = 1 ]; then
    WIT_STATUS=SKIPPED WIT_REASON="--skip-witness EMERGENCY ESCAPE (the pin-11 no-fire class is unguarded this run)"
  elif [ ! -x "$WIT_PY" ]; then
    WIT_STATUS=SKIP WIT_REASON="missing tt-llk venv ($WIT_PY)"
  elif [ ! -x "$WIT_CXX" ]; then
    WIT_STATUS=SKIP WIT_REASON="missing pinned SFPI toolchain ($WIT_CXX)"
  elif [ "$DRY_RUN" = 1 ]; then
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

  # Corpus compile gate: every mapped corpus row must COMPILE green on BH
  # with the pinned toolchain before any 2x2 phase runs.  Compile-mode only
  # (no simulator, no device), so it is safe before the flocked phases.  A
  # completed PASS gate for the same SWEEP_DATE is reused (idempotent
  # resume); --dry-run proves the wiring with a plan-only pass instead.
  GATE_STATUS="" GATE_REASON=""
  GATE_ROOT="$EV/corpus-compile-gate"
  GATE_PY="$HERE/../python_tests/.venv/bin/python"
  GATE_CXX="$HERE/../sfpi/compiler/bin/riscv-tt-elf-g++"
  GATE_CMD=(python3 "$HERE/sfpu_corpus.py" --mode compile --arch bh --execute --require-executed-mapped)
  if [ ! -x "$GATE_PY" ]; then
    GATE_STATUS=SKIP GATE_REASON="missing tt-llk venv ($GATE_PY)"
  elif [ ! -x "$GATE_CXX" ]; then
    GATE_STATUS=SKIP GATE_REASON="missing pinned SFPI toolchain ($GATE_CXX)"
  elif [ "$DRY_RUN" = 1 ]; then
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
    # sfpu_corpus.py refuses a pre-existing --run-root; rotate a stale one.
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
fi

if [ "$MODE" = weekly ]; then
  # No --schedule filter: the weekly deliberately runs EVERY ops.tsv row,
  # including the schedule=weekly deferrals the nightly skips (the
  # device-time budget split is data in the TSV, not a fork).
  DELTA+=(--knob-attribution --knob-silicon-rows "$HEADLINE_ROWS")
fi

# ------------------------------------------------------------ the sweep --
# RATIFIED (owner, 2026-08-20; charter §1(3) amended same day): sweeps run
# STRAIGHT SILICON — per-cell device-golden correctness legs gate every perf
# cell; CRAQ is a debug/lane-validation oracle (pinned sims), not a sweep
# precondition.  sweep_2x2.py records the CRAQ-gate taint/status either way.
# Unbuffered: without this the sweep's stdout block-buffers into the tee'd
# log (a setsid-detached sweep looks dead for minutes between flushes).
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
  ${DELTA[@]+"${DELTA[@]}"} \
  ${PREV:+--prev-run "$PREV"} \
  "$@"
RC=$?

# ------------------------------------------------------ mode post-gates --
if [ "$MODE" = nightly ] && [ -f "$EV/REPORT.md" ]; then
  # Append the gate self-test evidence so every nightly verdict carries the
  # proof that its flip detector works.
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

DEJAGNU_NOTE=""
if [ "$MODE" = weekly ]; then
  # DejaGnu byte-parity suites against the pinned toolchain build tree.  The
  # counting/gating logic lives in dejagnu_gate.sh (self-tested by
  # selftest_dejagnu_gate.sh: clean->GREEN, failing->RED, no-sum->RED); the
  # self-test runs first so a broken gate can never bless tonight's suites.
  bash "$HERE/selftest_dejagnu_gate.sh" > "$EV/selftest-dejagnu-gate.txt" 2>&1 \
    || { echo "RED: dejagnu gate self-test failed (see $EV/selftest-dejagnu-gate.txt)"; RC=1; }
  DEJAGNU_BUILD_TREE="$DEJAGNU_BUILD_TREE" SFPI_GCC_SRC="$SFPI_GCC_SRC" \
    DEJAGNU_SUITES="$DEJAGNU_SUITES" bash "$HERE/dejagnu_gate.sh" "$EV" "${1:-}" || RC=1
  DEJAGNU_NOTE="; dejagnu: $EV/dejagnu/summary.txt"
fi

echo "== $MODE sweep $DATE done rc=$RC; report: $EV/REPORT.md$DEJAGNU_NOTE =="
exit $RC
