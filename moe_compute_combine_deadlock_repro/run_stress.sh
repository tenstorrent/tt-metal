#!/bin/bash
# Run INSIDE the container. Single device-open session, SMOKE_ITERS iterations of
# dispatch -> fused moe_compute -> sync with NO reset between iters. Detects a mid-loop
# hang via progress-stall (no new log output for STALL_S while the process is alive) and
# reports the iteration it stalled on. Strict hygiene at the end.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/.env.sh"

ITERS="${ITERS:-100}"
STALL_S="${STALL_S:-150}"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$HERE/logs/stress_${TS}.txt"
echo "iters=$ITERS  stall_thresh=${STALL_S}s  mux=${SMOKE_MUX:-default(3,0,4,7)}  seed=${SMOKE_SEED:-random}"
echo "log: $LOG"

# Launch the stress run; retry on the intermittent post-reset fabric-init FATAL
# (set_fabric_config) which would otherwise kill the run before iter 1.
launch() { SMOKE_ITERS="$ITERS" stdbuf -oL -eL python3 moe_compute_smoke.py >"$LOG" 2>&1 & PID=$!; }
launch
echo "smoke PID: $PID"
for _try in 1 2 3; do
    sleep 8
    if grep -q "could not fit in the discovered physical topology" "$LOG" 2>/dev/null; then
        echo "fabric-init flake (attempt $_try); relaunching (NOTE: cannot host-reset from container)"
        kill -9 "$PID" 2>/dev/null; sleep 2; launch; echo "relaunched PID: $PID"
    else
        break
    fi
done

VERDICT="UNKNOWN"
last_size=0; last_change=$(date +%s)
while kill -0 "$PID" 2>/dev/null; do
    sleep 10
    if grep -q "SMOKE STRESS PASSED" "$LOG"; then VERDICT="PASS"; break; fi
    if grep -qiE "Traceback|RuntimeError|TT_FATAL|critical" "$LOG"; then VERDICT="ERROR"; break; fi
    cur_size=$(wc -c <"$LOG" 2>/dev/null || echo 0)
    now=$(date +%s)
    if [ "$cur_size" != "$last_size" ]; then last_size=$cur_size; last_change=$now; fi
    if [ $((now - last_change)) -ge "$STALL_S" ]; then VERDICT="HANG"; break; fi
done
if ! kill -0 "$PID" 2>/dev/null; then
    grep -q "SMOKE STRESS PASSED" "$LOG" && VERDICT="PASS" || { [ "$VERDICT" = UNKNOWN ] && VERDICT="EXITED"; }
fi
echo ">>> STRESS VERDICT: $VERDICT"
echo "last progress: $(grep -E 'STRESS iter|STRESS PASSED' "$LOG" | tail -1)"
[ "$VERDICT" = "HANG" -o "$VERDICT" = "ERROR" ] && { echo '--- log tail ---'; tail -20 "$LOG"; }

# --- strict hygiene ---
kill -TERM "$PID" 2>/dev/null
for i in $(seq 1 10); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
kill -KILL "$PID" 2>/dev/null
sleep 2
pkill -9 -f moe_compute_smoke.py 2>/dev/null; pkill -9 -f tt-triage 2>/dev/null; sleep 1
echo "=== straggler check ==="
ps -eo pid,cmd | grep -E "moe_compute_smoke|tt-triage" | grep -v grep && echo "WARN stragglers" || echo "no stragglers"
H=0; for d in /dev/tenstorrent/*; do fuser "$d" >/dev/null 2>&1 && { echo "HELD $d"; H=1; }; done
[ "$H" = 0 ] && echo "no device fds held" || echo "WARN device fds held"
rm -f /dev/shm/TT_UMD_LOCK.* /dev/shm/tt_device_*_memory
echo ">>> STRESS DONE verdict=$VERDICT (host should tt-smi -r before next run)"
