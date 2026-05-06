#!/usr/bin/env bash
# Race scenario: Process A (hang app) vs Process B (open_and_close).
#
# Expected sequence:
#   A starts, acquires SafeDeviceGuard, dispatches hanging kernel.
#   B starts 3s later, blocks on the SafeDeviceGuard mutex.
#   A's timeout fires → on_hang() marks dirty + runs tt-smi -r, exception propagates.
#   Hang app catches the exception and calls _Exit(0) — mutex abandoned, dirty=1 persists.
#   B acquires the abandoned mutex (EOWNERDEAD recovery), sees dirty=1, runs tt-smi -r.
#   B opens the device cleanly and exits 0.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../../../../.build/default"

HANG_APP="${BUILD_DIR}/tools/tests/triage/hang_apps/add_2_integers_hang/Release/triage_hang_app_add_2_integers_hang"
OPEN_CLOSE="${BUILD_DIR}/tools/tests/triage/hang_apps/open_and_close/Release/triage_open_and_close"

echo "=== SafeDeviceGuard race test ==="
echo "A: $HANG_APP"
echo "B: $OPEN_CLOSE"
echo ""

export TT_METAL_SAFE_DEVICE_OPEN=1
# 15s timeout: enough for A to acquire the device and start the workload but not run forever.
export TT_METAL_OPERATION_TIMEOUT_SECONDS=15

# Launch A first.  After 3 s it will have acquired the SafeDeviceGuard and started the
# hanging workload.  Then launch B, which will block on the mutex until A exits.
echo "--- Launching A (hang app) ---"
"$HANG_APP" >"$SCRIPT_DIR/A.log" 2>&1 &
PID_A=$!
echo "A pid=$PID_A"

echo "--- Sleeping 3s so A can acquire the lock before B starts ---"
sleep 3

echo "--- Launching B (open_and_close) ---"
"$OPEN_CLOSE" >"$SCRIPT_DIR/B.log" 2>&1 &
PID_B=$!
echo "B pid=$PID_B"

wait "$PID_A"; RC_A=$?
wait "$PID_B"; RC_B=$?

echo ""
echo "=== A log ==="
cat "$SCRIPT_DIR/A.log"
echo ""
echo "=== B log ==="
cat "$SCRIPT_DIR/B.log"
echo ""
echo "=== Results ==="
echo "A exit code: $RC_A"
echo "B exit code: $RC_B"

PASS=1

if [[ $RC_A -ne 0 ]]; then
    echo "FAIL: A exited with $RC_A (expected 0)"
    PASS=0
fi
if [[ $RC_B -ne 0 ]]; then
    echo "FAIL: B exited with $RC_B (expected 0)"
    PASS=0
fi

# A must have hit the timeout path.
if ! grep -q "Timeout detected\|device timeout" "$SCRIPT_DIR/A.log"; then
    echo "FAIL: A log missing timeout signal"
    PASS=0
fi

# SafeDeviceGuard::on_hang() must have fired.
if ! grep -q "SafeDeviceGuard: mesh dirty" "$SCRIPT_DIR/A.log"; then
    echo "FAIL: A log missing SafeDeviceGuard on_hang signal"
    PASS=0
fi

# B must have acquired the device.
if ! grep -q "device acquired" "$SCRIPT_DIR/B.log"; then
    echo "FAIL: B log missing 'device acquired'"
    PASS=0
fi

if [[ $PASS -eq 1 ]]; then
    echo "PASS"
else
    echo "FAIL"
    exit 1
fi
