#!/usr/bin/env bash
# Validates the SafeDeviceGuard recovery path without corrupt device state.
#
# Step 1: Start open_and_close as "A", let it open the device (dirty=1 set),
#         then SIGKILL it — mutex abandoned, dirty=1 persists, device is NOT corrupted.
#
# Step 2: B (another open_and_close) acquires the abandoned mutex, sees dirty=1,
#         runs tt-smi -r as a precaution, then opens and closes cleanly.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../../../../.build/default"
OPEN_CLOSE="${BUILD_DIR}/tools/tests/triage/hang_apps/open_and_close/Debug/triage_open_and_close"
HOLD_DEVICE="${BUILD_DIR}/tools/tests/triage/hang_apps/open_and_close/Debug/triage_hold_device"

export TT_METAL_SAFE_DEVICE_OPEN=1
export TT_METAL_OPERATION_TIMEOUT_SECONDS=60

echo "=== Step 1: A opens device, gets SIGKILL while holding SafeDeviceGuard ==="
"$HOLD_DEVICE" >"$SCRIPT_DIR/A.log" 2>&1 &
PID_A=$!
# Give A enough time to acquire SafeDeviceGuard and open the device, then kill it.
sleep 15
echo "Sending SIGKILL to A (pid=$PID_A)"
kill -KILL "$PID_A" 2>/dev/null || true
wait "$PID_A" 2>/dev/null || true
echo ""
echo "=== A log ==="
cat "$SCRIPT_DIR/A.log"
echo ""

# Confirm dirty bit is set.
DIRTY=$(cat /dev/shm/TT_METAL_DEVICE_DIRTY.mesh-0 2>/dev/null | od -An -tu1 | tr -d ' ' || echo "0")
echo "dirty byte after SIGKILL: $DIRTY"
if [[ "$DIRTY" != "1" ]]; then
    echo "FAIL: dirty bit should be 1 after SIGKILL (got $DIRTY)"
    exit 1
fi
echo "PASS: dirty bit is 1 as expected"
echo ""

echo "--- Sleeping 15s to let any UMD state settle ---"
sleep 15

echo ""
echo "=== Step 2: B recovers (EOWNERDEAD + dirty=1 → tt-smi -r → clean open) ==="
"$OPEN_CLOSE" >"$SCRIPT_DIR/B.log" 2>&1
RC_B=$?

echo "=== B log ==="
cat "$SCRIPT_DIR/B.log"
echo ""
echo "B exit code: $RC_B"

PASS=1
if [[ $RC_B -ne 0 ]]; then
    echo "FAIL: B exited with $RC_B"
    PASS=0
fi
if ! grep -q "SafeDeviceGuard: mesh dirty" "$SCRIPT_DIR/B.log"; then
    echo "FAIL: B did not detect dirty state"
    PASS=0
fi
if ! grep -q "mesh reset complete" "$SCRIPT_DIR/B.log"; then
    echo "FAIL: B did not complete tt-smi reset"
    PASS=0
fi
if ! grep -q "device acquired" "$SCRIPT_DIR/B.log"; then
    echo "FAIL: B did not acquire device"
    PASS=0
fi

if [[ $PASS -eq 1 ]]; then
    echo "PASS"
else
    echo "FAIL"
    exit 1
fi
