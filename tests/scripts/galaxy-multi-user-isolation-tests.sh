#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

TRAY_PODS_DIR="/app/multihosttest"
REQUIRED_PODS=4
TIMEOUT=300

EPOCH_START=$(date +%s)
echo "start epoch: $EPOCH_START"

TRAY_POD_FILE="$TRAY_PODS_DIR/tray-presence-$(hostname).txt"
echo "Creating $TRAY_POD_FILE"
touch "$TRAY_POD_FILE"

# wait for REQUIRED .txt files to appear (with timeout)
while true; do
  files=( "$TRAY_PODS_DIR"/tray-presence-*.txt )
  count=${#files[@]}
  elapsed=$(( $(date +%s) - EPOCH_START))

  if [ "$count" -ge "$REQUIRED_PODS" ]; then
    break
  fi

  if [ "$elapsed" -ge "$TIMEOUT" ]; then
    echo "Error: Not all $REQUIRED_PODS files appeared within $TIMEOUT seconds (found $count)."
    exit 1
  fi

  sleep 1
done

echo "All $REQUIRED_PODS docker containers have created their files (found $count)."

# install the wheel
echo "Installing the wheel"
#pip install /app/dist/*.whl

# run tests but capture exit codes
EXIT_CODE=0

echo "=== Building test_system_health ==="
set +e
./build/test/tt_metal/tt_fabric/test_system_health --cluster-type T3K 2>&1 | tee test2.log
rc=$?
set -e
if [ $rc -ne 0 ]; then
  echo "build for T3K test_system_health failed (exit code $rc). See test2.log"
  EXIT_CODE=$rc
fi

echo "=== Running async test ==="
set +e
pytest -v "tt-metal/tests/nightly/t3000/ccl/test_minimal_all_gather_async.py" 2>&1 | tee test1.log
rc=$?
set -e
if [ $rc -ne 0 ]; then
  echo "test_minimal_all_gather_async.py has failed (exit code $rc). See test1.log"
  EXIT_CODE=$rc
fi

if [ $EXIT_CODE -eq 0 ]; then
  echo "All tests passed"
else
  echo "At least one test failed. Check individual logs for details."
fi

exit $EXIT_CODE
