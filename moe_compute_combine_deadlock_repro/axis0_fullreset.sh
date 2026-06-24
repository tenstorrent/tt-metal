#!/bin/bash
# Run on HOST. Rigorous control: give cluster_axis=0 the IDENTICAL full reset that made
# axis1 pass, applied IMMEDIATELY before the run:
#   1) docker restart (clears any wedged proc / container state)
#   2) tt-smi -glx_reset_auto (retrains galaxy ethernet/fabric)
#   3) cluster_axis=0 SMOKE_PINPOINT -> if it STILL hangs after a full container+machine
#      reset, the deadlock is a genuine topology/op bug, not a stale-state artifact.
set -u
C=tt-xla-ird-mvasiljev
D=/home/mvasiljev/tt-metal/moe_compute_combine_deadlock_repro
HOSTD=/data/mvasiljev/tt-metal/moe_compute_combine_deadlock_repro
TTSMI=/usr/local/bin/tt-smi
SEED="${SEED:-4242}"
DWELL="${DWELL:-60}"
TS=$(date +%Y%m%d_%H%M%S)
OUT="$HOSTD/logs/axis0_fullreset_${TS}.txt"
slog="$D/logs/axis0_fr_seed${SEED}_${TS}.txt"
hslog="$HOSTD/logs/axis0_fr_seed${SEED}_${TS}.txt"
tlog="$D/logs/axis0_fr_triage_${TS}.txt"
htlog="$HOSTD/logs/axis0_fr_triage_${TS}.txt"
tlog_cs="$D/logs/axis0_fr_triage_callstacks_${TS}.txt"
htlog_cs="$HOSTD/logs/axis0_fr_triage_callstacks_${TS}.txt"
log(){ echo "$@" | tee -a "$OUT"; }

log "=== axis0_fullreset seed=$SEED @ $TS (container restart + glx_reset_auto, immediately before) ==="
log "--- 1) docker restart $C ---"
timeout 120 docker restart "$C" >>"$OUT" 2>&1 && log "  container restarted" || log "  WARN restart failed"
sleep 3
log "--- 2) tt-smi -glx_reset_auto ---"
timeout 300 "$TTSMI" -glx_reset_auto >>"$OUT" 2>&1; log "  reset rc=$?"
timeout 60 "$TTSMI" -ls >/dev/null 2>&1 && log "  detectable" || log "  WARN undetectable"

log "--- 3) cluster_axis=0 seed=$SEED SMOKE_PINPOINT ---"
log "  smoke log: $hslog"
docker exec "$C" bash -lc "source $D/.env.sh && SMOKE_CLUSTER_AXIS=0 SMOKE_SEED=$SEED SMOKE_PINPOINT=1 stdbuf -oL python3 moe_compute_smoke.py >$slog 2>&1 & echo \$! >/tmp/smoke_ax0fr.pid; wait" &
DEXEC=$!
state="timeout"
for i in $(seq 1 110); do
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
  log "  combine enqueued; dwelling ${DWELL}s..."
  sleep "$DWELL"
  if grep -q "SMOKE SYNC after moe_compute combine OK" "$hslog" 2>/dev/null; then state="drained"; log "  -> sync RETURNED (drained)"
  else state="hung"; log "  -> CONFIRMED HUNG after full container+machine reset"; fi
fi
if [ "$state" = "hung" ]; then
  log "  running tt-triage..."
  docker exec "$C" bash -lc "source $D/.env.sh && timeout 200 python3 \$TT_METAL_RUNTIME_ROOT/tools/tt-triage.py --skip-version-check >$tlog 2>&1" 2>/dev/null
  docker exec "$C" bash -lc "source $D/.env.sh && timeout 300 python3 \$TT_METAL_RUNTIME_ROOT/tools/tt-triage.py --skip-version-check --run=dump_callstacks --all-cores -vv >$tlog_cs 2>&1" 2>/dev/null
  running=$(grep -E 'RUNNING.*MoEComputeDeviceOperation' "$htlog" 2>/dev/null | grep -oE '[0-9]+(, [0-9]+)*' | tail -1)
  halt=$(grep -oE 'Failed to halt .* on device [0-9]+' "$htlog" 2>/dev/null | grep -oE 'device [0-9]+' | sort -u | tr '\n' ' ')
  n352=$(grep -Ec 'writer.cpp[: ]+352' "$htlog_cs" 2>/dev/null)
  log "    RUNNING MoECompute devices : $running"
  log "    Failed-to-halt devices     : $halt"
  log "    cores at writer.cpp:352    : $n352"
  log "    triage log: $htlog"
  log "    callstack triage log: $htlog_cs"
fi
log "  --- smoke tail ---"; tail -10 "$hslog" 2>/dev/null | sed 's/^/    /' | tee -a "$OUT" >/dev/null
docker exec "$C" bash -lc "p=\$(cat /tmp/smoke_ax0fr.pid 2>/dev/null); [ -n \"\$p\" ] && kill -TERM \$p 2>/dev/null; sleep 4; [ -n \"\$p\" ] && kill -KILL \$p 2>/dev/null" 2>/dev/null
kill "$DEXEC" 2>/dev/null; wait "$DEXEC" 2>/dev/null
log "=== RESULT: cluster_axis=0 after full container+machine reset -> $state ==="
log "full log: $OUT"
