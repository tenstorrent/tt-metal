#!/bin/bash
# Run INSIDE the container. Device-open stress inducer (SMOKE_ITERS, no reset between iters).
# On a progress-stall, run tt-triage against the LIVE wedged runtime BEFORE killing anything,
# then check whether the captured signature matches the canonical combine deadlock
# (op MoEComputeDeviceOperation RUNNING on devices 16,20,24,28 at writer.cpp:365). Strict
# hygiene at the end. Prints machine-greppable result lines:
#   VERIFY VERDICT: <PASS|STALL|ERROR|EXITED>
#   VERIFY SIGMATCH: <YES|NO|n/a>   (writer.cpp:365 + devices 16,20,24,28 seen in triage)
#   VERIFY STALLPOINT: <text>       (how far the smoke got before going silent)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/.env.sh"

ITERS="${ITERS:-30}"
STALL_S="${STALL_S:-150}"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$HERE/logs/verify_stress_${TS}.txt"
TRIAGE_LOG="$HERE/logs/verify_triage_${TS}.txt"
echo "iters=$ITERS stall=${STALL_S}s mux=${SMOKE_MUX:-default(3,0,4,7)} seed=${SMOKE_SEED:-random}"
echo "stress log: $LOG"
echo "triage log: $TRIAGE_LOG"

launch() { SMOKE_ITERS="$ITERS" stdbuf -oL -eL python3 moe_compute_smoke.py >"$LOG" 2>&1 & PID=$!; }
launch
echo "smoke PID: $PID"
# tolerate the post-reset fabric-init flake
for _try in 1 2 3; do
    sleep 8
    if grep -q "could not fit in the discovered physical topology" "$LOG" 2>/dev/null; then
        echo "fabric-init flake (try $_try); relaunch"; kill -9 "$PID" 2>/dev/null; sleep 2; launch
    else break; fi
done

VERDICT="UNKNOWN"
last_size=0; last_change=$(date +%s)
while kill -0 "$PID" 2>/dev/null; do
    sleep 10
    if grep -q "SMOKE STRESS PASSED" "$LOG"; then VERDICT="PASS"; break; fi
    if grep -qiE "Traceback|RuntimeError|TT_FATAL|critical" "$LOG"; then VERDICT="ERROR"; break; fi
    cur_size=$(wc -c <"$LOG" 2>/dev/null || echo 0); now=$(date +%s)
    if [ "$cur_size" != "$last_size" ]; then last_size=$cur_size; last_change=$now; fi
    if [ $((now - last_change)) -ge "$STALL_S" ]; then VERDICT="STALL"; break; fi
done
if ! kill -0 "$PID" 2>/dev/null; then
    grep -q "SMOKE STRESS PASSED" "$LOG" && VERDICT="PASS" || { [ "$VERDICT" = UNKNOWN ] && VERDICT="EXITED"; }
fi

# Characterize how far it got (stall point)
okc=$(grep -c "moe_compute ok" "$LOG" 2>/dev/null || echo 0)
itc=$(grep -E "STRESS iter [0-9]+/" "$LOG" | tail -1)
STALLPOINT="moe_compute_ok=$okc last_iter='${itc:-none}'"

SIGMATCH="n/a"
if [ "$VERDICT" = "STALL" ]; then
    echo ">>> STALL detected — triaging LIVE wedged runtime (NOT killing yet)..."
    timeout 300 python3 "$TT_METAL_RUNTIME_ROOT/tools/tt-triage.py" --skip-version-check >"$TRIAGE_LOG" 2>&1
    echo "================ LIVE STALL SIGNATURE ================"
    grep -E "RUNNING|DeviceOperation|writer.cpp" "$TRIAGE_LOG" | grep -iE "running|combine|moe|writer.cpp" | head -15
    echo "====================================================="
    has_line=$(grep -c "writer.cpp 365" "$TRIAGE_LOG" 2>/dev/null || echo 0)
    has_dev=$(grep -c "16, 20, 24, 28" "$TRIAGE_LOG" 2>/dev/null || echo 0)
    has_op=$(grep -cE "MoEComputeDeviceOperation.*RUNNING|RUNNING.*MoEComputeDeviceOperation" "$TRIAGE_LOG" 2>/dev/null || echo 0)
    echo "triage hits: writer.cpp:365=$has_line  devices(16,20,24,28)=$has_dev  MoECompute RUNNING=$has_op"
    if [ "$has_line" -gt 0 ] && [ "$has_dev" -gt 0 ]; then SIGMATCH="YES"; else SIGMATCH="NO"; fi
fi

echo "VERIFY VERDICT: $VERDICT"
echo "VERIFY SIGMATCH: $SIGMATCH"
echo "VERIFY STALLPOINT: $STALLPOINT"

# --- strict hygiene ---
kill -TERM "$PID" 2>/dev/null
for i in $(seq 1 10); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
kill -KILL "$PID" 2>/dev/null; sleep 2
pkill -9 -f moe_compute_smoke.py 2>/dev/null; pkill -9 -f tt-triage 2>/dev/null; sleep 1
ps -eo pid,cmd | grep -E "moe_compute_smoke|tt-triage" | grep -v grep >/dev/null && echo "WARN stragglers" || echo "no stragglers"
H=0; for d in /dev/tenstorrent/*; do fuser "$d" >/dev/null 2>&1 && { echo "HELD $d"; H=1; }; done
[ "$H" = 0 ] && echo "no device fds held" || echo "WARN device fds held"
rm -f /dev/shm/TT_UMD_LOCK.* /dev/shm/tt_device_*_memory
echo ">>> VERIFY DONE (host should tt-smi -r before next run)"
