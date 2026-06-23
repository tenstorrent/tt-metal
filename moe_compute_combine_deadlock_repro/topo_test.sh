#!/bin/bash
# Run on HOST. cluster_axis=0 fused combine with an EXPLICIT topology (Ring or Linear) and
# live tt-triage, to check whether Linear hangs on the SAME 4 devices (16/20/24/28) as the
# default Ring capture. TOPO=linear|ring (default linear).
set -u
C=tt-xla-ird-mvasiljev
D=/home/mvasiljev/tt-metal/moe_compute_combine_deadlock_repro
HOSTD=/data/mvasiljev/tt-metal/moe_compute_combine_deadlock_repro
TTSMI=/usr/local/bin/tt-smi
SEED="${SEED:-4242}"
TOPO="${TOPO:-linear}"
DWELL="${DWELL:-60}"
TS=$(date +%Y%m%d_%H%M%S)
OUT="$HOSTD/logs/topo_${TOPO}_${TS}.txt"
slog="$D/logs/topo_${TOPO}_smoke_${TS}.txt"
hslog="$HOSTD/logs/topo_${TOPO}_smoke_${TS}.txt"
tlog="$D/logs/topo_${TOPO}_triage_${TS}.txt"
htlog="$HOSTD/logs/topo_${TOPO}_triage_${TS}.txt"
log(){ echo "$@" | tee -a "$OUT"; }

log "=== topo_test TOPO=$TOPO seed=$SEED @ $TS ==="
log "--- glx_reset_auto ---"; timeout 300 "$TTSMI" -glx_reset_auto >>"$OUT" 2>&1; log "  reset rc=$?"
timeout 60 "$TTSMI" -ls >/dev/null 2>&1 && log "  detectable" || log "  WARN undetectable"

log "--- cluster_axis=0 SMOKE_COMBINE_TOPO=$TOPO SMOKE_PINPOINT ---"
log "  smoke log: $hslog"
docker exec "$C" bash -lc "source $D/.env.sh && SMOKE_CLUSTER_AXIS=0 SMOKE_SEED=$SEED SMOKE_COMBINE_TOPO=$TOPO SMOKE_PINPOINT=1 stdbuf -oL python3 moe_compute_smoke.py >$slog 2>&1 & echo \$! >/tmp/smoke_topo.pid; wait" &
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
  log "  combine enqueued; dwelling ${DWELL}s..."; sleep "$DWELL"
  if grep -q "SMOKE SYNC after moe_compute combine OK" "$hslog" 2>/dev/null; then state="drained"; log "  -> sync RETURNED (drained)"
  else state="hung"; log "  -> CONFIRMED HUNG"; fi
fi
if [ "$state" = "hung" ]; then
  log "  running tt-triage..."
  docker exec "$C" bash -lc "source $D/.env.sh && timeout 200 python3 \$TT_METAL_RUNTIME_ROOT/tools/tt-triage.py --skip-version-check >$tlog 2>&1" 2>/dev/null
  running=$(grep -E 'RUNNING.*MoEComputeDeviceOperation' "$htlog" 2>/dev/null | head -1 | grep -oE '16, 20, 24, 28|[0-9]+(, [0-9]+){1,}' | head -1)
  halt=$(grep -oE 'Failed to halt .* on device [0-9]+' "$htlog" 2>/dev/null | grep -oE 'device [0-9]+' | sort -u | tr '\n' ' ')
  n352=$(grep -c 'writer.cpp 352' "$htlog" 2>/dev/null)
  log "    RUNNING MoECompute devices : $running"
  log "    Failed-to-halt devices     : $halt"
  log "    cores at writer.cpp:352    : $n352"
fi
log "  --- smoke tail ---"; tail -8 "$hslog" 2>/dev/null | sed 's/^/    /' | tee -a "$OUT" >/dev/null
docker exec "$C" bash -lc "p=\$(cat /tmp/smoke_topo.pid 2>/dev/null); [ -n \"\$p\" ] && kill -TERM \$p 2>/dev/null; sleep 4; [ -n \"\$p\" ] && kill -KILL \$p 2>/dev/null" 2>/dev/null
kill "$DEXEC" 2>/dev/null; wait "$DEXEC" 2>/dev/null
log "=== RESULT: TOPO=$TOPO -> $state  (devices: ${running:-n/a}) ==="
