#!/bin/bash
# Run INSIDE the container. Reproduce the hang, run default tt-triage against the LIVE
# runtime, and print the hung-op + device signature. Then strict hygiene (kill own PID,
# verify no stragglers / device fds, clean container /dev/shm). Operator resets on host.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/.env.sh"

TS=$(date +%Y%m%d_%H%M%S)
SMOKE_LOG="$HERE/logs/sigcheck_smoke_$TS.txt"
TRIAGE_LOG="$HERE/logs/sigcheck_triage_$TS.txt"
echo "runtime HEAD: $(cd "$TT_METAL_RUNTIME_ROOT" && git rev-parse --short HEAD)"
echo "smoke log:  $SMOKE_LOG"
echo "triage log: $TRIAGE_LOG"

SMOKE_PINPOINT=1 stdbuf -oL -eL python3 moe_compute_smoke.py >"$SMOKE_LOG" 2>&1 &
SMOKE_PID=$!
echo "smoke PID: $SMOKE_PID"

for i in $(seq 1 150); do
    grep -q "moe_compute ok" "$SMOKE_LOG" && break
    if ! kill -0 "$SMOKE_PID" 2>/dev/null; then echo "smoke exited early; see $SMOKE_LOG"; tail -8 "$SMOKE_LOG"; exit 1; fi
    sleep 2
done
echo "combine enqueued; 60s dwell to confirm hang..."
sleep 60
if grep -q "SMOKE SYNC after moe_compute combine OK" "$SMOKE_LOG"; then
    echo ">>> NOT HUNG this run (sync returned)."; kill -TERM "$SMOKE_PID" 2>/dev/null
else
    echo ">>> CONFIRMED HUNG. Running default tt-triage..."
    timeout 300 python3 "$TT_METAL_RUNTIME_ROOT/tools/tt-triage.py" --skip-version-check >"$TRIAGE_LOG" 2>&1
    echo "================ HUNG-OP SIGNATURE ================"
    grep -E "RUNNING|DeviceOperation" "$TRIAGE_LOG" | grep -iE "running|combine|moe" | head -20
    echo "=================================================="
fi

# --- strict hygiene ---
kill -TERM "$SMOKE_PID" 2>/dev/null
for i in $(seq 1 10); do kill -0 "$SMOKE_PID" 2>/dev/null || break; sleep 1; done
kill -KILL "$SMOKE_PID" 2>/dev/null
sleep 2
pkill -9 -f moe_compute_smoke.py 2>/dev/null
pkill -9 -f tt-triage 2>/dev/null
sleep 1
echo "=== straggler check ==="
ps -eo pid,cmd | grep -E "moe_compute_smoke|tt-triage" | grep -v grep && echo "WARN: stragglers remain" || echo "no stragglers"
echo "=== device fd holders ==="
H=0; for d in /dev/tenstorrent/*; do fuser "$d" >/dev/null 2>&1 && { echo "HELD: $d"; H=1; }; done
[ "$H" = 0 ] && echo "no device fds held" || echo "WARN: device fds held"
rm -f /dev/shm/TT_UMD_LOCK.* /dev/shm/tt_device_*_memory
ls -la /dev/shm 2>/dev/null | grep -iE "tt|umd|sem" || echo "shm clean"
echo ">>> DONE (host should tt-smi -r before next run)"
