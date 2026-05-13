#!/usr/bin/env bash
# Sequential recovery test: validates the "A crashed, B recovers" path.
#
# Step 1: A runs with TT_METAL_SAFE_DEVICE_OPEN=1 and hangs.
#         on_hang() fires, tt-smi -r runs, A exits via _Exit(0).
#         The SafeDeviceGuard mutex is abandoned and dirty=1 persists.
#
# Step 2: B runs with TT_METAL_SAFE_DEVICE_OPEN=1.
#         Acquires the abandoned mutex (EOWNERDEAD recovery), sees dirty=1,
#         runs tt-smi -r, then opens and closes the device cleanly.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../../../../.build/default"

HANG_APP="${BUILD_DIR}/tools/tests/triage/hang_apps/add_2_integers_hang/Debug/triage_hang_app_add_2_integers_hang"
OPEN_CLOSE="${BUILD_DIR}/tools/tests/triage/hang_apps/open_and_close/Debug/triage_open_and_close"

export TT_METAL_SAFE_DEVICE_OPEN=1
export TT_METAL_OPERATION_TIMEOUT_SECONDS=15

echo "=== Step 1: A (hang app) — expect timeout → on_hang() → _Exit(0) ==="
"$HANG_APP" >"$SCRIPT_DIR/A.log" 2>&1
RC_A=$?

echo "=== A log ==="
cat "$SCRIPT_DIR/A.log"
echo ""
echo "A exit code: $RC_A"
echo ""

PASS=1
if [[ $RC_A -ne 0 ]]; then
    echo "FAIL: A exited with $RC_A (expected 0)"
    PASS=0
fi
if ! grep -q "Timeout detected\|device timeout" "$SCRIPT_DIR/A.log"; then
    echo "FAIL: A log missing timeout signal"
    PASS=0
fi
if ! grep -q "SafeDeviceGuard: mesh dirty" "$SCRIPT_DIR/A.log"; then
    echo "FAIL: A log missing SafeDeviceGuard on_hang signal"
    PASS=0
fi
if ! grep -q "mesh reset complete" "$SCRIPT_DIR/A.log"; then
    echo "FAIL: A log missing tt-smi reset completion"
    PASS=0
fi

echo "--- Sleeping 20s to let the device settle after A's tt-smi -r reset ---"
sleep 20
echo ""
echo "=== Step 2: B (open_and_close) — expect EOWNERDEAD recovery, dirty check, clean run ==="
"$OPEN_CLOSE" >"$SCRIPT_DIR/B.log" 2>&1
RC_B=$?

echo "=== B log ==="
cat "$SCRIPT_DIR/B.log"
echo ""
echo "B exit code: $RC_B"
echo ""

if [[ $RC_B -ne 0 ]]; then
    echo "FAIL: B exited with $RC_B (expected 0)"
    PASS=0
fi
if ! grep -q "device acquired" "$SCRIPT_DIR/B.log"; then
    echo "FAIL: B log missing 'device acquired'"
    PASS=0
fi

echo "=== Results ==="
if [[ $PASS -eq 1 ]]; then
    echo "PASS"
else
    echo "FAIL"
    exit 1
fi
