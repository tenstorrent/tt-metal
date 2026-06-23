#!/bin/bash
# Reset the wedged galaxy (container restart + glx_reset_auto), then run the mesh probe.
set -u
C=tt-xla-ird-mvasiljev
D=/home/mvasiljev/tt-metal/moe_compute_combine_deadlock_repro
HOSTD=/data/mvasiljev/tt-metal/moe_compute_combine_deadlock_repro
TTSMI=/usr/local/bin/tt-smi
TS=$(date +%Y%m%d_%H%M%S)
OUT="$HOSTD/logs/probe_mesh_${TS}.txt"
slog="$D/logs/probe_mesh_run_${TS}.txt"
hslog="$HOSTD/logs/probe_mesh_run_${TS}.txt"
log(){ echo "$@" | tee -a "$OUT"; }

log "=== run_probe @ $TS ==="
log "--- docker restart $C ---"; timeout 120 docker restart "$C" >>"$OUT" 2>&1 && log "  restarted" || log "  WARN restart failed"
sleep 3
log "--- glx_reset_auto ---"; timeout 300 "$TTSMI" -glx_reset_auto >>"$OUT" 2>&1; log "  reset rc=$?"
timeout 60 "$TTSMI" -ls >/dev/null 2>&1 && log "  detectable" || log "  WARN undetectable"
log "--- probe_mesh.py ---"
docker exec "$C" bash -lc "source $D/.env.sh && timeout 180 python3 probe_mesh.py >$slog 2>&1; echo EXIT=\$?" | tee -a "$OUT"
log "--- probe output ---"
grep -E "get_device_ids|^R\\\\C|^R[0-9]|col[0-9] ring|row[0-9]:|dev [0-9]|stuck span|CCL plane|replicate group|NOT a single|Traceback|Error|FATAL" "$hslog" 2>/dev/null | sed 's/^/  /' | tee -a "$OUT"
log "full probe log: $hslog"
