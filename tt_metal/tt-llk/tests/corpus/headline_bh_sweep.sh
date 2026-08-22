#!/usr/bin/env bash
# Headline BH 2x2 sweep entry point — "show me the flips FAST" (owner
# priority order 2026-08-20: headline/targeted runs precede full-surface
# runs).  One command that measures ONLY the rows that can plausibly have
# flipped since the last pin:
#
#   ops = HEADLINE_ROWS (sweep_2x2.conf)
#       UNION every ops-TSV row whose fresh body / golden / mapped test
#         changed since the previous pin (derived from git by
#         headline_ops.py; see its stderr log in the evidence dir).
#
# Pass --ops a,b,c to override the derivation entirely (forwarded verbatim).
# All OTHER pins and gates are the weekly's: conf-lint (self-tested first),
# corpus validation, report/enforcement gate self-tests, pinned toolchain +
# sim shas, straight-silicon ratified default, baseline drift thresholds.
# Deliberately OMITTED vs the weekly, for speed: per-knob attribution legs
# and the DejaGnu byte-parity suites — run the weekly for those.
#
# Once sweep_2x2.py grows --priority-ops (cross-pin reuse lane), the conf's
# HEADLINE_ROWS are additionally passed there so the flip-prone rows are
# measured first; until then the wrapper says so and runs in config order.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

# --allow-pin-override: the ONLY sanctioned way to run with a PINNED_* value
# from the environment (sweep_2x2.conf rejects silent env overrides).  The
# flag is consumed here, never forwarded to sweep_2x2.py.
ARGS=()
for _a in "$@"; do
  if [ "$_a" = "--allow-pin-override" ]; then
    export ALLOW_PIN_OVERRIDE=1
    echo "headline: --allow-pin-override — environment pin values will be honored AND LOGGED"
  else
    ARGS+=("$_a")
  fi
done
set -- ${ARGS[@]+"${ARGS[@]}"}

# conf-lint FIRST (enforcement layer, ledger item 10), linter self-test
# before the linter — a broken linter can never bless a sweep.
bash "$HERE/selftest_conf_lint.sh" > /tmp/headline-selftest-conf-lint.$$ 2>&1 \
  || { echo "FATAL: conf-lint self-test failed:"; cat /tmp/headline-selftest-conf-lint.$$; rm -f /tmp/headline-selftest-conf-lint.$$; exit 2; }
bash "$HERE/conf_lint.sh" || { echo "FATAL: conf-lint refused — pin audit trail disagrees (fix conf prose/baseline header in the same commit as the pin change)"; exit 2; }

# shellcheck source=sweep_2x2.conf
source "$HERE/sweep_2x2.conf" || { echo "FATAL: sweep_2x2.conf refused (pin override without --allow-pin-override?)"; exit 2; }

DATE=${SWEEP_DATE:-$(date +%Y%m%d)}
EV="$EVIDENCE_ROOT/headline-$DATE"
BASELINE="$HERE/sfpu_device_baseline_${CHIP_CLASS}_v1.tsv"
[ -f "$BASELINE" ] || { echo "FATAL: no baseline for chip class '$CHIP_CLASS' ($BASELINE)"; exit 2; }
# KERNEL-scoped (v2) VERDICT baseline (lane ET, owner ratification
# 2026-08-21): passed when seeded; absent = bootstrap (kernel ratios report
# no-baseline, v1 diagnostic checks keep full severity — handover rule).
KBASELINE="$HERE/sfpu_device_baseline_${CHIP_CLASS}_v2.tsv"
[ -f "$KBASELINE" ] || KBASELINE=""

# Wrapper-lib self-test, then the evidence-root collision guard (incident
# 2026-08-20: pin-14 classify wrote 15 min into the pin-12 weekly-20260820
# root).  SWEEP_DATE stays the manual root-name override.
bash "$HERE/selftest_sweep_wrapper_lib.sh" > /tmp/headline-selftest-wrapper-lib.$$ 2>&1 \
  || { echo "FATAL: sweep_wrapper_lib self-test failed:"; cat /tmp/headline-selftest-wrapper-lib.$$; rm -f /tmp/headline-selftest-wrapper-lib.$$; exit 2; }
# shellcheck source=sweep_wrapper_lib.sh
source "$HERE/sweep_wrapper_lib.sh" || { echo "FATAL: sweep_wrapper_lib.sh missing/broken"; exit 2; }
evidence_root_guard "$EV" "$PINNED_CC1PLUS_SHA256" "headline_bh_sweep.sh" || exit 3

# --prev-run chain: newest N clean run roots, consumed twice by
# sweep_2x2.py: scoreboard drift annotation (newest root) + cross-pin
# cell reuse (every root probed; source roots provenance-gated at
# adoption time — markers, pin record, craq-gate taint parity).
PREV=$(newest_clean_runs "$EVIDENCE_ROOT" "$EV" "${SWEEP_PREV_CHAIN:-3}" headline weekly nightly)

echo "== headline sweep $DATE -> $EV (prev chain: ${PREV:-none}) =="

python3 "$HERE/sfpu_corpus.py" --validate || { echo "FATAL: corpus validation failed"; exit 2; }

# Gate self-tests first (weekly parity): a broken gate must never bless a run.
mkdir -p "$EV"
mv /tmp/headline-selftest-wrapper-lib.$$ "$EV/selftest-wrapper-lib.txt" 2>/dev/null || true
python3 "$HERE/selftest_sweep_2x2_report.py" > "$EV/selftest-report-gate.txt" 2>&1 \
  || { echo "FATAL: report-gate self-test failed (see $EV/selftest-report-gate.txt)"; exit 2; }
python3 "$HERE/selftest_enforcement_gates.py" > "$EV/selftest-enforcement-gates.txt" 2>&1 \
  || { echo "FATAL: enforcement-gates self-test failed (see $EV/selftest-enforcement-gates.txt)"; exit 2; }
python3 "$HERE/selftest_knob_legs_semleg.py" > "$EV/selftest-knob-legs-semleg.txt" 2>&1 \
  || { echo "FATAL: knob-legs/sem-leg self-test failed (see $EV/selftest-knob-legs-semleg.txt)"; exit 2; }
python3 "$HERE/selftest_dst_layout_32b.py" > "$EV/selftest-dst-layout-32b.txt" 2>&1 \
  || { echo "FATAL: dst-layout-32b wiring self-test failed (see $EV/selftest-dst-layout-32b.txt)"; exit 2; }
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
mv /tmp/headline-selftest-conf-lint.$$ "$EV/selftest-conf-lint.txt" 2>/dev/null || true
bash "$HERE/conf_lint.sh" > "$EV/conf-lint.txt" 2>&1 \
  || { echo "FATAL: conf-lint refused (see $EV/conf-lint.txt)"; exit 2; }

# Ops list: explicit --ops wins; otherwise derive headline + changed-since-
# last-pin rows from git (derivation log kept in evidence).
HAVE_OPS=0
EXPLICIT_OPS=""
_prev=""
for _a in "$@"; do
  [ "$_prev" = "--ops" ] && EXPLICIT_OPS=$_a
  [ "$_a" = "--ops" ] && HAVE_OPS=1
  case "$_a" in --ops=*) HAVE_OPS=1; EXPLICIT_OPS=${_a#--ops=};; esac
  _prev=$_a
done
OPS_ARGS=()
if [ "$HAVE_OPS" = 1 ]; then
  echo "headline: explicit --ops passed — skipping git derivation"
else
  OPS=$(python3 "$HERE/headline_ops.py" --headline "$HEADLINE_ROWS" \
        2> "$EV/headline-ops-derivation.txt") \
    || { echo "FATAL: headline_ops.py failed:"; cat "$EV/headline-ops-derivation.txt"; exit 2; }
  echo "headline: ops = $OPS"
  echo "headline: derivation log: $EV/headline-ops-derivation.txt"
  OPS_ARGS=(--ops "$OPS")
fi

# --priority-ops: measure the flip-prone headline rows first, once the
# harness supports it (lane DC's cross-pin reuse work); harmless until then.
# With an explicit --ops list, only the headline rows INSIDE that list are
# passed — sweep_2x2.py refuses any --priority-ops row outside --ops
# ("unknown ops", rc=1) AFTER the evidence root is stamped, contaminating
# the root (lane FI/FM launch gotcha, harness-audit FO fix).
PRIO_ARGS=()
if grep -q -- '"--priority-ops"' "$HERE/sweep_2x2.py"; then
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
    PRIO_ARGS=(--priority-ops "$PRIO_ROWS")
    echo "headline: --priority-ops supported — priority rows: $PRIO_ROWS"
  else
    echo "headline: explicit --ops shares no rows with HEADLINE_ROWS — no --priority-ops"
  fi
else
  echo "headline: sweep_2x2.py has no --priority-ops yet — rows run in config order"
fi

# RATIFIED (owner, 2026-08-20; charter §1(3) amended same day): sweeps run
# STRAIGHT SILICON — per-cell device-golden correctness legs gate every perf
# cell; CRAQ is a debug/lane-validation oracle (pinned sims), not a sweep
# precondition.  sweep_2x2.py records the CRAQ-gate taint/status either way.
# Unbuffered: without this the sweep's stdout block-buffers into the tee'd
# log (a setsid-detached sweep looks dead for many minutes between flushes).
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
  ${OPS_ARGS[@]+"${OPS_ARGS[@]}"} \
  ${PRIO_ARGS[@]+"${PRIO_ARGS[@]}"} \
  ${PREV:+--prev-run "$PREV"} \
  "$@"
RC=$?

echo "== headline sweep $DATE done rc=$RC; report: $EV/REPORT.md =="
exit $RC
