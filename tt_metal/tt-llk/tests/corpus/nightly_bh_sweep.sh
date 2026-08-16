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
# shellcheck source=sweep_2x2.conf
source "$HERE/sweep_2x2.conf"

DATE=${SWEEP_DATE:-$(date +%Y%m%d)}
EV="$EVIDENCE_ROOT/nightly-$DATE"
BASELINE="$HERE/sfpu_device_baseline_${CHIP_CLASS}_v1.tsv"
[ -f "$BASELINE" ] || { echo "FATAL: no baseline for chip class '$CHIP_CLASS' ($BASELINE)"; exit 2; }

# Previous nightly run for before/after drift (newest dated dir that is not us).
PREV=$(ls -d "$EVIDENCE_ROOT"/nightly-* 2>/dev/null | grep -vx "$EV" | sort | tail -1 || true)

echo "== nightly sweep $DATE -> $EV (prev: ${PREV:-none}) =="

# Corpus inventory must be self-consistent before any measurement.
python3 "$HERE/sfpu_corpus.py" --validate || { echo "FATAL: corpus validation failed"; exit 2; }

# sweep_2x2.py preflight enforces: compiler sha256 == pin, removed loadmacro
# flags error on use, OFF/ON flag sets accepted. It classifies BEFORE any
# device job, refuses byte-identical pairs, gates silicon on paired CRAQ,
# serializes every device job under both flocks, copies CSVs in-lock, and
# never parses results while a lock is held. Baselines are read-only here.
python3 "$HERE/sweep_2x2.py" \
  --evidence-root "$EV" \
  --compiler-sha "$PINNED_COMPILER_SHA256" \
  --sim-bh "$SIM_BH" --sim-wh "$SIM_WH" \
  --allow-hardware \
  --baseline "$BASELINE" \
  --max-drift-pct "$MAX_DRIFT_PCT" \
  ${PREV:+--prev-run "$PREV"} \
  "$@"
RC=$?

echo "== nightly sweep $DATE done rc=$RC; report: $EV/REPORT.md =="
exit $RC
