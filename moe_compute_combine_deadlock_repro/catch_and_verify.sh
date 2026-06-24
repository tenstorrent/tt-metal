#!/bin/bash
# Run on HOST. Watcher that verifies the NEXT stall instead of assuming it.
# Per attempt: reset+settle -> device-open stress inducer (verify_stall.sh) which, on a
# progress-stall, triages the LIVE wedged runtime and reports whether the signature matches
# the canonical combine deadlock. If a STALL is caught, we then reset+settle and run the
# EXACT unmodified single-shot test (run_triage_sig.sh) as a cross-check: does the regime
# reproduce the confirmed hang on the canonical path, or was the stress stall harness-specific?
# Stops as soon as a STALL is caught and cross-checked (or after ATTEMPTS passing runs).
set -u
C=tt-xla-ird-mvasiljev
D=/home/mvasiljev/tt-metal/moe_compute_combine_deadlock_repro
HD=/data/mvasiljev/tt-metal/moe_compute_combine_deadlock_repro
ATTEMPTS="${ATTEMPTS:-10}"; SETTLE="${SETTLE:-30}"; ITERS="${ITERS:-30}"
TS=$(date +%Y%m%d_%H%M%S)
SUM="$HD/logs/catch_verify_${TS}.txt"
echo "=== catch_and_verify @ $TS attempts=$ATTEMPTS settle=${SETTLE}s iters=$ITERS ===" | tee "$SUM"

for a in $(seq 1 "$ATTEMPTS"); do
  echo "" | tee -a "$SUM"; echo "########## attempt $a/$ATTEMPTS ##########" | tee -a "$SUM"
  tt-smi -r >/dev/null 2>&1; echo "  reset rc=$? ; settle ${SETTLE}s" | tee -a "$SUM"; sleep "$SETTLE"

  out=$(docker exec --user 4123:4123 "$C" /bin/bash -lc "ITERS=$ITERS bash $D/verify_stall.sh" 2>&1)
  v=$(echo "$out" | grep -oE 'VERIFY VERDICT: [A-Z_]+' | awk '{print $3}' | head -1)
  sm=$(echo "$out" | grep -oE 'VERIFY SIGMATCH: [A-Za-z/]+' | awk '{print $3}' | head -1)
  sp=$(echo "$out" | grep 'VERIFY STALLPOINT:' | head -1)
  echo "  inducer -> verdict=$v sigmatch=$sm  [$sp]" | tee -a "$SUM"

  if [ "$v" = "STALL" ]; then
    echo "  >>> STALL CAUGHT. Live-triage signature match = $sm" | tee -a "$SUM"
    echo "$out" | grep -E "triage hits:|LIVE STALL SIGNATURE|writer.cpp|16, 20, 24, 28|MoEComputeDeviceOperation" | head -12 | sed 's/^/      /' | tee -a "$SUM"
    # cross-check with the EXACT unmodified single-shot test
    echo "  --- cross-check: reset+settle, run canonical single-shot (run_triage_sig.sh) ---" | tee -a "$SUM"
    tt-smi -r >/dev/null 2>&1; sleep "$SETTLE"
    cx=$(docker exec --user 4123:4123 "$C" /bin/bash -lc "bash $D/run_triage_sig.sh" 2>&1)
    if echo "$cx" | grep -qi "CONFIRMED HUNG"; then
      echo "  >>> SINGLE-SHOT ALSO HUNG (regime genuinely bad). signature:" | tee -a "$SUM"
      echo "$cx" | grep -E "writer.cpp 365|16, 20, 24, 28|MoEComputeDeviceOperation" | head -6 | sed 's/^/      /' | tee -a "$SUM"
      echo "  RESULT: CONFIRMED_BAD_REGIME (stress sigmatch=$sm, single-shot=HANG)" | tee -a "$SUM"
    else
      echo "  >>> SINGLE-SHOT PASSED (stress stall likely harness/device-open-specific, not the canonical deadlock)" | tee -a "$SUM"
      echo "  RESULT: STRESS_STALL_ONLY (stress sigmatch=$sm, single-shot=PASS)" | tee -a "$SUM"
    fi
    tt-smi -r >/dev/null 2>&1
    echo ">>> CATCH_VERIFY DONE (caught+verified on attempt $a) -> $SUM" | tee -a "$SUM"
    exit 0
  fi
done

echo "" | tee -a "$SUM"; echo "=== final reset ===" | tee -a "$SUM"; tt-smi -r >/dev/null 2>&1; echo "reset rc=$?" | tee -a "$SUM"
echo ">>> CATCH_VERIFY DONE (no stall caught in $ATTEMPTS attempts; device in good window) -> $SUM" | tee -a "$SUM"
