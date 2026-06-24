#!/bin/bash
# Run on HOST. Reset, then run cluster_axis=1 (8-device col ring) with SMOKE_PINPOINT.
# Expect the post-combine sync to RETURN (drains, per TUNING_LOG on build 68e82deb155).
# If it hangs, run tt-triage to capture the stuck chips for comparison with axis0.
set -u
C=tt-xla-ird-mvasiljev
D=/home/mvasiljev/tt-metal/moe_compute_combine_deadlock_repro
HOSTD=/data/mvasiljev/tt-metal/moe_compute_combine_deadlock_repro
TTSMI=/usr/local/bin/tt-smi
SEED="${SEED:-4242}"
DWELL="${DWELL:-60}"
TS=$(date +%Y%m%d_%H%M%S)
OUT="$HOSTD/logs/axis1_only_${TS}.txt"
slog="$D/logs/axis1_seed${SEED}_${TS}.txt"
hslog="$HOSTD/logs/axis1_seed${SEED}_${TS}.txt"
tlog="$D/logs/axis1_triage_${TS}.txt"
tlog_cs="$D/logs/axis1_triage_callstacks_${TS}.txt"
log(){ echo "$@" | tee -a "$OUT"; }

log "=== axis1_only seed=$SEED @ $TS ==="
# tt-smi -r leaves the galaxy fabric intermittently untrained (open_mesh_device fails at
# topology_mapper "target node not mapped"); glx_reset_auto retrains ethernet -> robust.
log "--- tt-smi -glx_reset_auto ---"
timeout 300 "$TTSMI" -glx_reset_auto >>"$OUT" 2>&1; log "  reset rc=$?"
timeout 60 "$TTSMI" -ls >/dev/null 2>&1 && log "  detectable" || log "  WARN undetectable"

log "########## cluster_axis=1 seed=$SEED ##########"
log "  smoke log: $hslog"
docker exec "$C" bash -lc "source $D/.env.sh && SMOKE_CLUSTER_AXIS=1 SMOKE_SEED=$SEED SMOKE_PINPOINT=1 stdbuf -oL python3 moe_compute_smoke.py >$slog 2>&1 & echo \$! >/tmp/smoke_ax1.pid; wait" &
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
  else state="hung"; log "  -> HUNG"; fi
fi
if [ "$state" = "hung" ]; then
  log "  running tt-triage..."
  docker exec "$C" bash -lc "source $D/.env.sh && timeout 200 python3 \$TT_METAL_RUNTIME_ROOT/tools/tt-triage.py --skip-version-check >$tlog 2>&1" 2>/dev/null
  docker exec "$C" bash -lc "source $D/.env.sh && timeout 300 python3 \$TT_METAL_RUNTIME_ROOT/tools/tt-triage.py --skip-version-check --run=dump_callstacks --all-cores -vv >$tlog_cs 2>&1" 2>/dev/null
  running=$(grep -E 'RUNNING.*MoEComputeDeviceOperation' "$HOSTD/logs/axis1_triage_${TS}.txt" 2>/dev/null | grep -oE '[0-9]+(, [0-9]+)*' | tail -1)
  halt=$(grep -oE 'Failed to halt .* on device [0-9]+' "$HOSTD/logs/axis1_triage_${TS}.txt" 2>/dev/null | grep -oE 'device [0-9]+' | sort -u | tr '\n' ' ')
  n352=$(grep -Ec 'writer.cpp[: ]+352' "$HOSTD/logs/axis1_triage_callstacks_${TS}.txt" 2>/dev/null)
  log "    RUNNING MoECompute devices : $running"
  log "    Failed-to-halt devices     : $halt"
  log "    cores at writer.cpp:352    : $n352"
  log "    triage log                 : $HOSTD/logs/axis1_triage_${TS}.txt"
  log "    callstack triage log       : $HOSTD/logs/axis1_triage_callstacks_${TS}.txt"
fi
log "  --- smoke tail ---"; tail -10 "$hslog" 2>/dev/null | sed 's/^/    /' | tee -a "$OUT" >/dev/null
docker exec "$C" bash -lc "p=\$(cat /tmp/smoke_ax1.pid 2>/dev/null); [ -n \"\$p\" ] && kill -TERM \$p 2>/dev/null; sleep 4; [ -n \"\$p\" ] && kill -KILL \$p 2>/dev/null" 2>/dev/null
kill "$DEXEC" 2>/dev/null; wait "$DEXEC" 2>/dev/null
log "=== RESULT: cluster_axis=1 -> $state ==="
log "full log: $OUT"
