#!/bin/bash
# Run INSIDE the container. Runs amorrisonTT's e2e glm_47 decode test (the one that
# PASSES on their machine) on OUR machine to separate machine-vs-config. Detects a hang
# faster than the 900s pytest-timeout, then does strict hygiene. Operator resets on host.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/.env.sh"
export PYTHONPATH="$RT:$PYTHONPATH"   # need repo root for the `models` package

ITERS="${E2E_ITERS:-3}"
KSEL="${E2E_KSEL:-8x4 and fabric_1D_ring}"
TMO="${E2E_TMO:-600}"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$HERE/logs/e2e_${TS}.txt"
echo "runtime HEAD: $(cd "$RT" && git rev-parse --short HEAD)"
echo "test:   models/common/tests/modules/moe/test_tt_moe_decode.py -k '$KSEL'"
echo "iters:  $ITERS   wrapper-timeout: ${TMO}s"
echo "log:    $LOG"

cd "$RT"
timeout "$TMO" python3 -m pytest -q \
    "models/common/tests/modules/moe/test_tt_moe_decode.py" \
    -k "$KSEL" -p no:randomly -s >"$LOG" 2>&1 &
PYPID=$!

VERDICT="UNKNOWN"
START=$(date +%s)
# Do NOT classify from transient log strings (startup logs contain benign "error"/"warning").
# Let pytest run to completion under its own `timeout`, then classify from the final summary.
LAST=""
while kill -0 "$PYPID" 2>/dev/null; do
    sleep 10
    NOW=$(date +%s); EL=$((NOW-START))
    CUR=$(tail -1 "$LOG" 2>/dev/null | cut -c1-90)
    [ "$CUR" != "$LAST" ] && { echo "  [${EL}s] $CUR"; LAST="$CUR"; }
done
wait "$PYPID"; RC=$?
if   grep -qE "[0-9]+ passed" "$LOG"; then VERDICT="PASS"
elif grep -qE "[0-9]+ failed" "$LOG"; then VERDICT="FAIL"
elif [ "$RC" = 124 ];                 then VERDICT="TIMEOUT_HANG"
else VERDICT="EXIT_$RC"; fi
echo ">>> VERDICT: $VERDICT"
echo "---- pytest summary tail ----"; tail -25 "$LOG"

# --- strict hygiene ---
kill -TERM "$PYPID" 2>/dev/null
for i in $(seq 1 10); do kill -0 "$PYPID" 2>/dev/null || break; sleep 1; done
kill -KILL "$PYPID" 2>/dev/null
sleep 2
pkill -9 -f test_tt_moe_decode 2>/dev/null
pkill -9 -f "pytest" 2>/dev/null
pkill -9 -f tt-triage 2>/dev/null
sleep 1
echo "=== straggler check ==="
ps -eo pid,cmd | grep -E "test_tt_moe_decode|pytest|moe_compute_smoke|tt-triage" | grep -v grep && echo "WARN: stragglers remain" || echo "no stragglers"
echo "=== device fd holders ==="
H=0; for d in /dev/tenstorrent/*; do fuser "$d" >/dev/null 2>&1 && { echo "HELD: $d"; H=1; }; done
[ "$H" = 0 ] && echo "no device fds held" || echo "WARN: device fds held"
rm -f /dev/shm/TT_UMD_LOCK.* /dev/shm/tt_device_*_memory
ls -la /dev/shm 2>/dev/null | grep -iE "tt|umd|sem" || echo "shm clean"
echo ">>> DONE verdict=$VERDICT (host should tt-smi -r before next run)"
