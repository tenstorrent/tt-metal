#!/bin/bash
# Run on HOST. Repeatedly attempt the 100-iter no-reset stress: reset+settle, run, and
# record the outcome (how many iters completed before PASS/HANG). Retries up to ATTEMPTS
# times to (a) get past the intermittent iter-1 hang and (b) see if a started loop ever
# deadlocks mid-run. Resets between ATTEMPTS only (NOT between the 100 iters).
set -u
C=tt-xla-ird-mvasiljev
D=/home/mvasiljev/tt-metal/moe_compute_combine_deadlock_repro
ATTEMPTS="${ATTEMPTS:-5}"; ITERS="${ITERS:-100}"; SETTLE="${SETTLE:-30}"
TS=$(date +%Y%m%d_%H%M%S)
SUM="/data/mvasiljev/tt-metal/moe_compute_combine_deadlock_repro/logs/stress_until_${TS}.txt"
echo "=== stress_until @ $TS ITERS=$ITERS ATTEMPTS=$ATTEMPTS ===" | tee "$SUM"
for a in $(seq 1 "$ATTEMPTS"); do
  echo "" | tee -a "$SUM"; echo "########## attempt $a/$ATTEMPTS ##########" | tee -a "$SUM"
  tt-smi -r >/dev/null 2>&1; echo "  reset rc=$? ; settle ${SETTLE}s" | tee -a "$SUM"; sleep "$SETTLE"
  out=$(docker exec --user 4123:4123 "$C" /bin/bash -lc "ITERS=$ITERS bash $D/run_stress.sh" 2>&1)
  verdict=$(echo "$out" | grep -oE 'STRESS VERDICT: [A-Z_]+' | head -1 | sed 's/STRESS VERDICT: //')
  # how many iters completed (from the smoke log referenced in run_stress stdout)
  slog=$(echo "$out" | grep -oE '/home/[^ ]*/stress_[0-9_]+\.txt' | head -1)
  lastiter=$(docker exec --user 4123:4123 "$C" /bin/bash -lc "grep -oE 'STRESS iter [0-9]+/' '$slog' 2>/dev/null | tail -1")
  passed=$(echo "$out" | grep -oE 'STRESS PASSED: [0-9]+/[0-9]+' | head -1)
  fabric=""; echo "$out" | grep -q "could not fit in the discovered physical topology" && fabric="(fabric-flake seen)"
  echo "  -> verdict=$verdict last='${lastiter:-none}' ${passed} $fabric" | tee -a "$SUM"
  if [ "$verdict" = "PASS" ]; then echo "  >>> COMPLETED $ITERS ITERS, stopping." | tee -a "$SUM"; break; fi
done
echo "" | tee -a "$SUM"; echo "=== final reset ==="; tt-smi -r >/dev/null 2>&1; echo "reset rc=$?" | tee -a "$SUM"
echo ">>> STRESS_UNTIL DONE -> $SUM" | tee -a "$SUM"
