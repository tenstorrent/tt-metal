#!/bin/bash
# Run INSIDE the container. Determines HANG vs PASS for the fused moe_compute combine
# smoke, keeping device hygiene tight: kills its own PID, waits for death, verifies no
# stragglers, and cleans the container /dev/shm UMD leftovers. Does NOT reset (operator
# does tt-smi -r on the host between runs).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/.env.sh"

LABEL="${1:-verdict}"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$HERE/logs/${LABEL}_${TS}.txt"
echo "label:       $LABEL"
echo "runtime HEAD: $(cd "$TT_METAL_RUNTIME_ROOT" && git rev-parse --short HEAD)"
echo "smoke log:   $LOG"
echo "TT_METAL_FORCE_JIT_COMPILE=${TT_METAL_FORCE_JIT_COMPILE:-unset}"

SMOKE_PINPOINT=1 stdbuf -oL -eL python3 moe_compute_smoke.py >"$LOG" 2>&1 &
SMOKE_PID=$!
echo "smoke PID: $SMOKE_PID"

# Wait for the combine to be enqueued (line right before the hanging sync).
# Generous window: a cold JIT compile of the whole graph can take many minutes.
ENQ=0
for i in $(seq 1 420); do
    grep -q "moe_compute ok" "$LOG" && { ENQ=1; break; }
    if ! kill -0 "$SMOKE_PID" 2>/dev/null; then echo "VERDICT: EARLY_EXIT (see log)"; tail -15 "$LOG"; break; fi
    sleep 2
done

VERDICT="UNKNOWN"
if [ "$ENQ" = 1 ]; then
    echo "combine enqueued; dwelling 60s to see if the post-combine sync returns..."
    for i in $(seq 1 30); do
        if grep -q "SMOKE SYNC after moe_compute combine OK" "$LOG"; then VERDICT="PASS"; break; fi
        if ! kill -0 "$SMOKE_PID" 2>/dev/null; then VERDICT="EXITED"; break; fi
        sleep 2
    done
    [ "$VERDICT" = "UNKNOWN" ] && VERDICT="HANG"
elif kill -0 "$SMOKE_PID" 2>/dev/null; then
    VERDICT="NO_ENQUEUE_BUT_ALIVE"
fi
echo ">>> VERDICT: $VERDICT"

# --- hygiene: ensure NO stragglers before we hand back for reset ---
kill -TERM "$SMOKE_PID" 2>/dev/null
for i in $(seq 1 10); do kill -0 "$SMOKE_PID" 2>/dev/null || break; sleep 1; done
kill -KILL "$SMOKE_PID" 2>/dev/null
sleep 2
# Kill any straggler smoke/triage just in case (this container is single-user for this work).
pkill -9 -f moe_compute_smoke.py 2>/dev/null
pkill -9 -f tt-triage 2>/dev/null
sleep 1
echo "=== straggler check ==="
ps -eo pid,cmd | grep -E "moe_compute_smoke|tt-triage" | grep -v grep && echo "WARN: stragglers remain" || echo "no stragglers"
echo "=== device fd holders ==="
H=0; for d in /dev/tenstorrent/*; do fuser "$d" >/dev/null 2>&1 && { echo "HELD: $d"; H=1; }; done
[ "$H" = 0 ] && echo "no device fds held" || echo "WARN: device fds held"
echo "=== clean container /dev/shm UMD leftovers ==="
rm -f /dev/shm/TT_UMD_LOCK.* /dev/shm/tt_device_*_memory
ls -la /dev/shm 2>/dev/null | grep -iE "tt|umd|sem" || echo "shm clean"
echo "tail of smoke log:"; tail -8 "$LOG"
echo ">>> DONE label=$LABEL verdict=$VERDICT (host should tt-smi -r before next run)"
