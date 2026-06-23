#!/bin/bash
# Run on the HOST. Like cycle_seed_test.sh, but on each HANG it keeps the smoke process
# ALIVE and runs tt-triage IN PARALLEL, then extracts a compact "stuck signature":
#   - the devices on which MoEComputeDeviceOperation is still RUNNING
#   - the devices/cores reporting "Failed to halt"
#   - the count of combine cores parked at writer.cpp:352 (final ring barrier)
# so we can see whether the SAME chips/cores are problematic every run.
# Then it kills only its own smoke PID and resets the galaxy (full completion).
set -u
C=tt-xla-ird-mvasiljev
D=/home/mvasiljev/tt-metal/moe_compute_combine_deadlock_repro
HOSTD=/data/mvasiljev/tt-metal/moe_compute_combine_deadlock_repro
RT=/home/mvasiljev/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal
SEEDS="${SEEDS:-1234 7 99}"
ITERS="${ITERS:-2}"
DWELL="${DWELL:-50}"            # seconds to confirm the post-moe_compute sync is hung
TTSMI=/usr/local/bin/tt-smi
TS=$(date +%Y%m%d_%H%M%S)
OUT="$HOSTD/logs/cycle_triage_${TS}.txt"
SIG="$HOSTD/logs/cycle_triage_${TS}_signatures.txt"

log(){ echo "$@" | tee -a "$OUT"; }
log "=== cycle_seed_triage seeds=[$SEEDS] iters=$ITERS dwell=${DWELL}s @ $TS ==="
log "runtime: $(docker exec $C bash -lc "cd $RT && git rev-parse --short HEAD" 2>/dev/null)"

reset_galaxy(){
  log "--- galaxy reset (letting it complete) ---"
  timeout 240 "$TTSMI" -glx_reset_auto >>"$OUT" 2>&1; local rc=$?
  timeout 60 "$TTSMI" -ls >/dev/null 2>&1 && log "    reset rc=$rc; galaxy detectable" || log "    reset rc=$rc; WARN not detectable"
}

run_one(){  # args: seed iter
  local seed="$1" iter="$2"
  local slog="$D/logs/triage_smoke_${seed}_i${iter}_${TS}.txt"
  local tlog="$D/logs/triage_dump_${seed}_i${iter}_${TS}.txt"
  log ""
  log "########## seed=$seed iter=$iter / $ITERS ##########"
  # 1) launch seeded smoke in background (NO timeout -> stays alive when hung)
  docker exec "$C" bash -lc "source $D/.env.sh && SMOKE_SEED=$seed SMOKE_PINPOINT=1 stdbuf -oL python3 moe_compute_smoke.py >$slog 2>&1 & echo \$! >/tmp/smoke_${seed}_${iter}.pid; wait" &
  local DEXEC=$!
  # 2) wait for combine to be enqueued
  for s in $(seq 1 120); do
    docker exec "$C" bash -lc "grep -q 'moe_compute ok' $slog" 2>/dev/null && break
    docker exec "$C" bash -lc "grep -qE 'SMOKE PASSED|Traceback' $slog" 2>/dev/null && break
    sleep 2
  done
  # 3) dwell to confirm the sync hangs
  sleep "$DWELL"
  if docker exec "$C" bash -lc "grep -q 'SMOKE SYNC after moe_compute combine OK' $slog" 2>/dev/null; then
    log "  result: NOT-HUNG (sync returned / drained this run)"
    SIGN["$seed,$iter"]="DRAINED"
  elif docker exec "$C" bash -lc "grep -qE 'SMOKE PASSED|Traceback' $slog" 2>/dev/null; then
    log "  result: smoke exited early (pass or error) -- see $slog"
    SIGN["$seed,$iter"]="EXITED"
  else
    log "  result: HUNG -> running tt-triage in parallel"
    docker exec "$C" bash -lc "source $D/.env.sh && timeout 200 python3 \$TT_METAL_RUNTIME_ROOT/tools/tt-triage.py --skip-version-check >$tlog 2>&1" 2>/dev/null
    # extract stuck signature
    local running halt n352
    running=$(docker exec "$C" bash -lc "grep -E 'RUNNING.*MoEComputeDeviceOperation' $tlog | grep -oE '[0-9]+(, [0-9]+)*' | tail -1" 2>/dev/null)
    halt=$(docker exec "$C" bash -lc "grep -oE 'Failed to halt .* on device [0-9]+' $tlog | grep -oE 'device [0-9]+' | sort -u | tr '\n' ' '" 2>/dev/null)
    n352=$(docker exec "$C" bash -lc "grep -c 'writer.cpp 352' $tlog" 2>/dev/null)
    log "    RUNNING MoECompute devices : $running"
    log "    Failed-to-halt devices     : $halt"
    log "    cores at writer.cpp:352    : $n352"
    SIGN["$seed,$iter"]="RUNNING=[$running] HALT=[$halt] w352=$n352"
  fi
  # 4) kill only our smoke
  docker exec "$C" bash -lc "p=\$(cat /tmp/smoke_${seed}_${iter}.pid 2>/dev/null); [ -n \"\$p\" ] && kill -TERM \$p 2>/dev/null; sleep 4; [ -n \"\$p\" ] && kill -KILL \$p 2>/dev/null" 2>/dev/null
  kill "$DEXEC" 2>/dev/null
  wait "$DEXEC" 2>/dev/null
}

declare -A SIGN
for seed in $SEEDS; do
  for it in $(seq 1 "$ITERS"); do
    run_one "$seed" "$it"
    reset_galaxy
  done
done

log ""
log "=== STUCK-SIGNATURE COMPARISON ==="
{ echo "seed,iter,signature"; for k in "${!SIGN[@]}"; do echo "$k => ${SIGN[$k]}"; done | sort; } | tee -a "$SIG" | tee -a "$OUT"
log "signatures: $SIG"
