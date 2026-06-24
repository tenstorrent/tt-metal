#!/bin/bash
# Run on the HOST. Compares the moe_compute fused-combine behaviour across dispatch rings:
#   1) tt-smi -r (reset all, as requested)
#   2) cluster_axis=0 (4-device row ring) with a NEW seed, SMOKE_PINPOINT -> expect HANG;
#      keep the smoke alive and run tt-triage in parallel to capture the stuck chips.
#   3) kill smoke, RESTART the container (clears the wedged D-state proc), tt-smi -r.
#   4) cluster_axis=1 (8-device col ring), SMOKE_PINPOINT -> expect the post-combine sync
#      to RETURN (drains). If it hangs instead, run tt-triage to capture those chips too.
set -u
C=tt-xla-ird-mvasiljev
D=/home/mvasiljev/tt-metal/moe_compute_combine_deadlock_repro          # container path
HOSTD=/data/mvasiljev/tt-metal/moe_compute_combine_deadlock_repro      # host path (logs readable here)
RT=/home/mvasiljev/tt-metal
TTSMI=/usr/local/bin/tt-smi
SEED="${SEED:-4242}"
DWELL="${DWELL:-60}"            # seconds to confirm the post-combine sync is hung
TS=$(date +%Y%m%d_%H%M%S)
OUT="$HOSTD/logs/axis_compare_${TS}.txt"

log(){ echo "$@" | tee -a "$OUT"; }

reset_all(){
  log "--- tt-smi -r (reset all) ---"
  timeout 240 "$TTSMI" -r >>"$OUT" 2>&1; local rc=$?
  if timeout 60 "$TTSMI" -ls >/dev/null 2>&1; then log "    reset rc=$rc; devices detectable"; else
    log "    reset rc=$rc; -r failed/undetectable -> falling back to glx_reset_auto"
    timeout 240 "$TTSMI" -glx_reset_auto >>"$OUT" 2>&1
    timeout 60 "$TTSMI" -ls >/dev/null 2>&1 && log "    glx_reset_auto ok; detectable" || log "    WARN still undetectable"
  fi
}

# run_axis <axis> <expect: hang|drain>
run_axis(){
  local axis="$1" expect="$2"
  local slog="$D/logs/axis${axis}_seed${SEED}_${TS}.txt"
  local hslog="$HOSTD/logs/axis${axis}_seed${SEED}_${TS}.txt"
  local tlog="$D/logs/axis${axis}_triage_${TS}.txt"
  local tlog_cs="$D/logs/axis${axis}_triage_callstacks_${TS}.txt"
  local htlog_cs="$HOSTD/logs/axis${axis}_triage_callstacks_${TS}.txt"
  log ""
  log "########## cluster_axis=$axis  seed=$SEED  (expect: $expect) ##########"
  log "  smoke log: $hslog"
  docker exec "$C" bash -lc "source $D/.env.sh && SMOKE_CLUSTER_AXIS=$axis SMOKE_SEED=$SEED SMOKE_PINPOINT=1 stdbuf -oL python3 moe_compute_smoke.py >$slog 2>&1 & echo \$! >/tmp/smoke_ax${axis}.pid; wait" &
  local DEXEC=$!

  # wait (<=200s) for a decisive marker
  local i state="timeout"
  for i in $(seq 1 100); do
    if   grep -q "SMOKE SYNC after moe_compute combine OK" "$hslog" 2>/dev/null; then state="drained"; break
    elif grep -qE "SMOKE PASSED" "$hslog" 2>/dev/null;                          then state="passed";  break
    elif grep -q "Traceback" "$hslog" 2>/dev/null;                               then state="error";   break
    elif grep -q "moe_compute ok" "$hslog" 2>/dev/null;                          then state="enqueued"; break
    fi
    kill -0 "$DEXEC" 2>/dev/null || { state="exited"; break; }
    sleep 2
  done
  log "  marker: $state"

  if [ "$state" = "enqueued" ]; then
    log "  combine enqueued; dwelling ${DWELL}s to confirm hang..."
    sleep "$DWELL"
    if grep -q "SMOKE SYNC after moe_compute combine OK" "$hslog" 2>/dev/null; then
      state="drained"; log "  -> sync RETURNED after dwell (drained)"
    else
      state="hung"; log "  -> CONFIRMED HUNG (post-combine sync did not return)"
    fi
  fi

  if [ "$state" = "hung" ]; then
    log "  running tt-triage in parallel..."
    docker exec "$C" bash -lc "source $D/.env.sh && timeout 200 python3 \$TT_METAL_RUNTIME_ROOT/tools/tt-triage.py --skip-version-check >$tlog 2>&1" 2>/dev/null
    docker exec "$C" bash -lc "source $D/.env.sh && timeout 300 python3 \$TT_METAL_RUNTIME_ROOT/tools/tt-triage.py --skip-version-check --run=dump_callstacks --all-cores -vv >$tlog_cs 2>&1" 2>/dev/null
    local running halt n352
    running=$(grep -E 'RUNNING.*MoEComputeDeviceOperation' "$HOSTD/logs/axis${axis}_triage_${TS}.txt" 2>/dev/null | grep -oE '[0-9]+(, [0-9]+)*' | tail -1)
    halt=$(grep -oE 'Failed to halt .* on device [0-9]+' "$HOSTD/logs/axis${axis}_triage_${TS}.txt" 2>/dev/null | grep -oE 'device [0-9]+' | sort -u | tr '\n' ' ')
    n352=$(grep -Ec 'writer.cpp[: ]+352' "$htlog_cs" 2>/dev/null)
    log "    RUNNING MoECompute devices : $running"
    log "    Failed-to-halt devices     : $halt"
    log "    cores at writer.cpp:352    : $n352"
    log "    triage log: $HOSTD/logs/axis${axis}_triage_${TS}.txt"
    log "    callstack triage log: $htlog_cs"
  fi

  # tail the smoke log for context
  log "  --- smoke tail ---"
  tail -8 "$hslog" 2>/dev/null | sed 's/^/    /' | tee -a "$OUT" >/dev/null

  # kill only this smoke
  docker exec "$C" bash -lc "p=\$(cat /tmp/smoke_ax${axis}.pid 2>/dev/null); [ -n \"\$p\" ] && kill -TERM \$p 2>/dev/null; sleep 4; [ -n \"\$p\" ] && kill -KILL \$p 2>/dev/null" 2>/dev/null
  kill "$DEXEC" 2>/dev/null; wait "$DEXEC" 2>/dev/null

  echo "$state"  # not logged (captured by caller via file)
  echo "$state" >"$HOSTD/logs/.axis${axis}_state_${TS}"
}

log "=== axis_compare seed=$SEED dwell=${DWELL}s @ $TS ==="
log "runtime: $(docker exec $C bash -lc "cd $RT && git rev-parse --short HEAD" 2>/dev/null)"

reset_all
run_axis 0 hang

log ""
log "=== restarting container to clear any wedged D-state proc, then reset ==="
timeout 120 docker restart "$C" >>"$OUT" 2>&1 && log "  container restarted" || log "  WARN restart failed"
sleep 3
reset_all
run_axis 1 drain

log ""
log "=== AXIS COMPARISON SUMMARY ==="
log "  cluster_axis=0 (4-dev ring): $(cat "$HOSTD/logs/.axis0_state_${TS}" 2>/dev/null)"
log "  cluster_axis=1 (8-dev ring): $(cat "$HOSTD/logs/.axis1_state_${TS}" 2>/dev/null)"
log "  full log: $OUT"
