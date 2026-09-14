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
# The host's tt-smi is a venv console script, not on PATH. Its shebang points at
# the venv python, so the absolute path needs no activation.
HOST_TT_SMI=/opt/tt_metal_infra/provisioning/provisioning_env/bin/tt-smi
# Bound on pytest's SIGINT teardown, and on the post-reset run so a device that
# never comes back cannot burn the whole step budget and lose the survivors' results.
STOP_TIMEOUT_SECS=120
POST_RESET_RUN_TIMEOUT_SECS=900

RESULTS_DIR=".multi-user-test-results"
RESET_DONE_FLAG="${RESULTS_DIR}/reset_done"
CONTAINER0_EXITED_FLAG="${RESULTS_DIR}/container0_pre_reset_exited"
C0_STOP_FLAG="${RESULTS_DIR}/container0_stop"
mkdir -p "$RESULTS_DIR"
rm -f "$RESET_DONE_FLAG" "$CONTAINER0_EXITED_FLAG" "$C0_STOP_FLAG"

# Prefix every line so the output of N parallel containers stays attributable.
run_in_container() {
    local container="$1" tmo="${2:-}"
    local cmd="pytest -v ${TEST_PATH} ${TEST_ARGS}"
    [ -n "$tmo" ] && cmd="timeout ${tmo} ${cmd}"
    docker exec "$container" bash -c "$cmd" 2>&1 | sed -u "s/^/[${container}] /"
}

# Run the workload in a loop until the reset_done flag appears, then run once
# more to confirm the container survived the reset. Writes exit status on completion.
run_survivor_loop() {
    local container="$1"
    (
        while [ ! -f "$RESET_DONE_FLAG" ]; do
            if ! run_in_container "$container"; then
                echo 1 > "${RESULTS_DIR}/${container}.status"
                exit 1
            fi
        done
        # Final run after reset to confirm ETH survived.
        rc=0
        run_in_container "$container" || rc=$?
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
    # Loop like the survivors: every suite but WH tray_reset's finishes inside
    # RESET_WAIT_SECS, leaving the reset nothing live to interrupt. `break` (not
    # `|| true`) so a genuine failure still reports via the flag.
    while [ ! -f "$C0_STOP_FLAG" ]; do
        run_in_container "$container0" || break
    done
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
    # SIGINT, not the default SIGTERM: CPython installs no SIGTERM handler, so
    # pytest would die instantly without fixture teardown, leaving the eth/fabric
    # firmware live on a board we are about to reset. SIGINT raises
    # KeyboardInterrupt, which tears the session down and closes the device.
    echo ">>> Stopping workload in ${container0} (SIGINT)..."
    touch "$C0_STOP_FLAG"
    docker exec "$container0" pkill -INT -f pytest || true

    # Wait for the teardown rather than guessing at it.
    for _ in $(seq "$STOP_TIMEOUT_SECS"); do
        docker exec "$container0" pgrep -f pytest >/dev/null 2>&1 || break
        sleep 1
    done
    if docker exec "$container0" pgrep -f pytest >/dev/null 2>&1; then
        echo ">>> WARNING: pytest still running after ${STOP_TIMEOUT_SECS}s; sending SIGKILL."
        echo ">>>          The device will NOT have been closed cleanly."
        docker exec "$container0" pkill -KILL -f pytest || true
        sleep 2
    fi

    # Collect the initial container-0 job (ignore its exit status — it was killed).
    wait "$container0_initial_pid" || true

    # Reset from the host, not via docker exec: the host tt-smi is newer than the
    # dev image's, and it sees all 32 boards, so device-node targets are
    # unambiguous (in-container, ids are relative to that container's view).
    # No fallback: -r is the only reset taking a device list, and the galaxy
    # resets would reset the survivor trays too.
    declare -a reset_targets=()
    for id in ${RESET_DEVICE_IDS//,/ }; do
        reset_targets+=("/dev/tenstorrent/${id}")
    done
    echo ">>> Resetting ${reset_targets[*]} via tt-smi -r ..."
    "$HOST_TT_SMI" --version || true
    if ! timeout 120 "$HOST_TT_SMI" -r "${reset_targets[@]}"; then
        echo ">>> ERROR: 'tt-smi -r ${reset_targets[*]}' failed or timed out on the host."
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
        run_in_container "$container0" "$POST_RESET_RUN_TIMEOUT_SECS" || rc=$?
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
