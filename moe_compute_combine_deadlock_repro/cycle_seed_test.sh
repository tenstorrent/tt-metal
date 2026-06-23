#!/bin/bash
# Run on the HOST. amorrison's consistency test: fixed seeds, cycle iterations to see
# whether the fused-combine hang reproduces every time (input-independent) or only
# sometimes, and whether different seeds (different routing/tokens) change it.
# Each iteration runs the seeded pinpoint smoke in the container under a timeout; a hang
# (exit 124) wedges the galaxy, so we reset with `tt-smi -r` between runs and let the
# reset FULLY COMPLETE (no interrupt). The galaxy is reset up-front too (it may be wedged).
set -u
C=tt-xla-ird-mvasiljev
D=/home/mvasiljev/tt-metal/moe_compute_combine_deadlock_repro
HOSTD=/data/mvasiljev/tt-metal/moe_compute_combine_deadlock_repro
SEEDS="${SEEDS:-1234 7 99}"
ITERS="${ITERS:-2}"
TMO="${TMO:-300}"
TS=$(date +%Y%m%d_%H%M%S)
OUT="$HOSTD/logs/cycle_seed_${TS}.txt"
TTSMI=/usr/local/bin/tt-smi

log(){ echo "$@" | tee -a "$OUT"; }

reset_galaxy(){
  log "--- tt-smi -r (letting it complete) ---"
  timeout 240 "$TTSMI" -r >>"$OUT" 2>&1
  local rc=$?
  log "--- reset rc=$rc ; verifying detectable ---"
  if timeout 60 "$TTSMI" -ls >/dev/null 2>&1; then log "    galaxy detectable"; else log "    WARN: galaxy NOT detectable after reset"; fi
}

run_one(){ # args: seed iter
  local seed=$1 iter=$2
  log ""
  log "########## seed=$seed iter=$iter / $ITERS ##########"
  local rc
  rc=$(docker exec "$C" bash -lc "source $D/.env.sh && SMOKE_SEED=$seed SMOKE_PINPOINT=1 timeout $TMO python3 moe_compute_smoke.py >/tmp/cyc_${seed}_${iter}.txt 2>&1; echo \${PIPESTATUS[0]:-\$?}" 2>/dev/null)
  local tail
  tail=$(docker exec "$C" bash -lc "grep -E 'SMOKE_SEED|moe_compute ok|SMOKE SYNC after moe_compute combine OK|SMOKE PASSED' /tmp/cyc_${seed}_${iter}.txt | tail -4" 2>/dev/null)
  log "exit_code=$rc"
  log "$tail"
  docker exec "$C" bash -lc "cp /tmp/cyc_${seed}_${iter}.txt $D/logs/cycle_${TS}_seed${seed}_iter${iter}.txt" 2>/dev/null
  if [ "$rc" = "124" ]; then echo "HANG"; else echo "PASS($rc)"; fi
}

log "=== cycle_seed_test seeds=[$SEEDS] iters=$ITERS timeout=${TMO}s reset='tt-smi -r' @ $TS ==="
declare -A RESULTS
FIRST=1
for seed in $SEEDS; do
  for i in $(seq 1 "$ITERS"); do
    reset_galaxy            # always start each run from a clean galaxy
    RESULTS["$seed/$i"]=$(run_one "$seed" "$i")
  done
done

log ""
log "=== SUMMARY ==="
for seed in $SEEDS; do
  for i in $(seq 1 "$ITERS"); do log "  seed=$seed iter=$i: ${RESULTS["$seed/$i"]}"; done
done
log "(galaxy left wedged if the last run hung -> reset before further use)"
