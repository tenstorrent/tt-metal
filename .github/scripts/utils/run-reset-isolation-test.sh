#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# run-reset-isolation-test.sh
#
# Orchestrate an ETH isolation test by:
#   1. Starting workloads in containers 1..N-1 in a loop; they keep running until
#      the reset completes, then do one final run to confirm ETH survived.
#   2. Starting a single workload run in container-0 (will be interrupted).
#   3. Waiting RESET_WAIT_SECS seconds.
#   4. Stopping container-0's workload, resetting its device(s) via tt-smi -r,
#      writing a "reset_done" flag that unblocks the survivor loop.
#   5. Restarting container-0's workload.
#   6. Verifying that all containers completed successfully.
#
# Usage:
#   bash run-reset-isolation-test.sh <NUM_CONTAINERS> <CONTAINER_PREFIX> \
#       <TEST_PATH> <TEST_ARGS> <RESET_DEVICE_IDS> [RESET_WAIT_SECS]
#
# Arguments:
#   NUM_CONTAINERS    Number of containers (e.g. 4 for tray_reset, 32 for asic_reset)
#   CONTAINER_PREFIX  Container name prefix (e.g. tray_reset, bh_tray_reset, chip_reset)
#   TEST_PATH         pytest path to run inside each container
#   TEST_ARGS         Additional pytest arguments (quoted string)
#   RESET_DEVICE_IDS  Comma-separated PCIe device IDs to reset via tt-smi -r
#   RESET_WAIT_SECS   Seconds to wait before triggering reset (default: 60)
#
# Exit code: always 0 — pass/fail determined by per-container .status files
# written to RESULTS_DIR, consumed by the caller's "Check test results" step.

set -euo pipefail

NUM_CONTAINERS="${1:?NUM_CONTAINERS required}"
CONTAINER_PREFIX="${2:?CONTAINER_PREFIX required}"
TEST_PATH="${3:?TEST_PATH required}"
TEST_ARGS="${4}"
RESET_DEVICE_IDS="${5:?RESET_DEVICE_IDS required}"
RESET_WAIT_SECS="${6:-60}"
# Matches ResetUtil.post_reset_settle_seconds in tests/sweep_framework/framework/tt_smi_util.py.
POST_RESET_SETTLE_SECS=10

RESULTS_DIR=".multi-user-test-results"
RESET_DONE_FLAG="${RESULTS_DIR}/reset_done"
CONTAINER0_EXITED_FLAG="${RESULTS_DIR}/container0_pre_reset_exited"
mkdir -p "$RESULTS_DIR"
rm -f "$RESET_DONE_FLAG" "$CONTAINER0_EXITED_FLAG"

# Run the workload in a loop until the reset_done flag appears, then run once
# more to confirm the container survived the reset. Writes exit status on completion.
run_survivor_loop() {
    local container="$1"
    (
        while [ ! -f "$RESET_DONE_FLAG" ]; do
            if ! docker exec "$container" bash -c "pytest -v ${TEST_PATH} ${TEST_ARGS}"; then
                echo 1 > "${RESULTS_DIR}/${container}.status"
                exit 1
            fi
        done
        # Final run after reset to confirm ETH survived.
        rc=0
        docker exec "$container" bash -c "pytest -v ${TEST_PATH} ${TEST_ARGS}" || rc=$?
        echo "$rc" > "${RESULTS_DIR}/${container}.status"
    ) &
}

echo "=== ETH Isolation Reset Test ==="
echo "  Containers  : ${NUM_CONTAINERS} (prefix: ${CONTAINER_PREFIX})"
echo "  Test        : ${TEST_PATH} ${TEST_ARGS}"
echo "  Reset IDs   : ${RESET_DEVICE_IDS}"
echo "  Reset delay : ${RESET_WAIT_SECS}s"
echo ""

# --- Step 1: Start survivor loop in containers 1..N-1 ---
declare -a bg_pids=()
for i in $(seq 1 $(( NUM_CONTAINERS - 1 ))); do
    container="${CONTAINER_PREFIX}-${i}"
    echo ">>> Starting survivor loop in ${container}"
    run_survivor_loop "$container"
    bg_pids+=($!)
done

# --- Step 2: Start workload in container-0 (will be interrupted and restarted) ---
container0="${CONTAINER_PREFIX}-0"
echo ">>> Starting workload in ${container0}"
(
    # `|| true`: set -e would otherwise skip the flag when pytest fails.
    docker exec "$container0" bash -c "pytest -v ${TEST_PATH} ${TEST_ARGS}" || true
    touch "$CONTAINER0_EXITED_FLAG"
) &
container0_initial_pid=$!

# --- Step 3: Wait, then reset container-0's device(s) ---
echo ">>> Waiting ${RESET_WAIT_SECS}s before reset..."
sleep "$RESET_WAIT_SECS"

reset_ok=1
if [ -f "$CONTAINER0_EXITED_FLAG" ]; then
    # Not a skip: a post-reset-only green would not have tested interruption.
    echo ">>> ERROR: ${container0}'s pre-reset workload exited within ${RESET_WAIT_SECS}s;"
    echo ">>>        there is no active workload for the reset to interrupt."
    reset_ok=0
else
    echo ">>> Stopping workload in ${container0}..."
    docker exec "$container0" pkill -f pytest || true

    # Give the process time to terminate and release device handles.
    sleep 5

    # Collect the initial container-0 job (ignore its exit status — it was killed).
    wait "$container0_initial_pid" || true

    # Whether -r takes these tray-mapping ids or its own per-container
    # enumeration is untested on WH, where the two differ. Log both to settle it.
    echo ">>> ${container0} holds /dev/tenstorrent/: $(docker exec "$container0" bash -c 'ls /dev/tenstorrent | sort -n | paste -sd,')"

    # No fallback: -r is the only reset taking a device list, and the galaxy
    # resets would reset the survivor trays too.
    echo ">>> Resetting device(s): ${RESET_DEVICE_IDS} via tt-smi -r ..."
    # timeout belongs inside: docker exec does not forward signals, so an outer
    # timeout would kill the client and leave tt-smi running.
    if ! docker exec "${container0}" timeout 120 tt-smi -r "$RESET_DEVICE_IDS"; then
        echo ">>> ERROR: 'tt-smi -r ${RESET_DEVICE_IDS}' failed or timed out in ${container0}."
        reset_ok=0
    else
        # Without this the confirming runs can fail on re-enumeration, not ETH state.
        echo ">>> Reset complete. Settling ${POST_RESET_SETTLE_SECS}s..."
        sleep "$POST_RESET_SETTLE_SECS"
    fi
fi

# Unblock the survivor loops so they perform their final confirming run.
touch "$RESET_DONE_FLAG"

# --- Step 4: Restart workload in container-0 ---
if [ "$reset_ok" = "1" ]; then
    echo ">>> Restarting workload in ${container0}..."
    (
        rc=0
        docker exec "$container0" bash -c "pytest -v ${TEST_PATH} ${TEST_ARGS}" || rc=$?
        echo "$rc" > "${RESULTS_DIR}/${container0}.status"
    ) &
    bg_pids+=($!)
else
    # No reset happened, so container-0 has nothing to confirm.
    echo 1 > "${RESULTS_DIR}/${container0}.status"
fi

# --- Step 5: Wait for all background jobs ---
echo ">>> Waiting for all containers to finish..."
for pid in "${bg_pids[@]}"; do
    wait "$pid" || true
done

echo ">>> All containers finished. Results in ${RESULTS_DIR}/"
# Exit 0: the caller's "Check test results" step reads .status files.
exit 0
