#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# run-reset-isolation-test.sh
#
# Orchestrate an ETH isolation test:
#   1. Start one long workload per container (pytest --count, so the device is
#      opened once and the run spans the reset).
#   2. After RESET_WAIT_SECS, stop container-0's workload and reset its devices.
#   3. Restart container-0's workload.
#   4. Every other container must have run through the reset undisturbed.
#
# Usage:
#   bash run-reset-isolation-test.sh <NUM_CONTAINERS> <CONTAINER_PREFIX> \
#       <TEST_PATH> <TEST_ARGS> <RESET_DEVICE_IDS> <REPEAT_COUNT> [RESET_WAIT_SECS]
#
# Exit code: always 0 — pass/fail determined by per-container .status files
# written to RESULTS_DIR, consumed by the caller's "Check test results" step.

set -euo pipefail

NUM_CONTAINERS="${1:?NUM_CONTAINERS required}"
CONTAINER_PREFIX="${2:?CONTAINER_PREFIX required}"
TEST_PATH="${3:?TEST_PATH required}"
TEST_ARGS="${4}"
RESET_DEVICE_IDS="${5:?RESET_DEVICE_IDS required}"
# Repetitions inside one pytest process, sized so the run outlasts the reset.
REPEAT_COUNT="${6:?REPEAT_COUNT required}"
RESET_WAIT_SECS="${7:-60}"

# Matches ResetUtil.post_reset_settle_seconds in tests/sweep_framework/framework/tt_smi_util.py.
POST_RESET_SETTLE_SECS=10
# Bound on pytest's SIGINT teardown, and on the post-reset run so a device that
# never comes back cannot burn the whole step budget and lose the other results.
STOP_TIMEOUT_SECS=300
POST_RESET_RUN_TIMEOUT_SECS=900
# The host's tt-smi is a venv console script, not on PATH. Its shebang points at
# the venv python, so the absolute path needs no activation.
HOST_TT_SMI=/opt/tt_metal_infra/provisioning/provisioning_env/bin/tt-smi

RESULTS_DIR=".multi-user-test-results"
mkdir -p "$RESULTS_DIR"
rm -f "${RESULTS_DIR}"/*.status

container0="${CONTAINER_PREFIX}-0"

# No -v: with REPEAT_COUNT repetitions across every container, per-test lines
# swamp the log. Prefix each line so parallel container output stays attributable.
run_in_container() {
    local container="$1" count="$2" tmo="${3:-}"
    local cmd="pytest ${TEST_PATH} ${TEST_ARGS} --count=${count}"
    [ -n "$tmo" ] && cmd="timeout ${tmo} ${cmd}"
    docker exec "$container" bash -c "$cmd" 2>&1 | sed -u "s/^/[${container}] /"
}

pytest_running() {
    docker exec "$1" pgrep -f pytest >/dev/null 2>&1
}

echo "=== ETH Isolation Reset Test ==="
echo "  Containers  : ${NUM_CONTAINERS} (prefix: ${CONTAINER_PREFIX})"
echo "  Test        : ${TEST_PATH} ${TEST_ARGS} --count=${REPEAT_COUNT}"
echo "  Reset IDs   : ${RESET_DEVICE_IDS}"
echo "  Reset delay : ${RESET_WAIT_SECS}s"
echo ""

# --- Step 1: One long run per container, all started together ---
declare -a bg_pids=()
for i in $(seq 1 $(( NUM_CONTAINERS - 1 ))); do
    container="${CONTAINER_PREFIX}-${i}"
    echo ">>> Starting workload in ${container}"
    (
        rc=0
        run_in_container "$container" "$REPEAT_COUNT" || rc=$?
        echo "$rc" > "${RESULTS_DIR}/${container}.status"
    ) &
    bg_pids+=($!)
done

echo ">>> Starting workload in ${container0} (to be interrupted)"
(
    run_in_container "$container0" "$REPEAT_COUNT" || true
) &
container0_initial_pid=$!

# --- Step 2: Wait, then reset container-0's device(s) ---
echo ">>> Waiting ${RESET_WAIT_SECS}s before reset..."
sleep "$RESET_WAIT_SECS"

# The reset must land while every container is mid-run, or it interrupts nothing
# and the survivors never had a workload to protect.
reset_ok=1
if ! pytest_running "$container0"; then
    echo ">>> ERROR: ${container0}'s workload is not running ${RESET_WAIT_SECS}s in;"
    echo ">>>        there is nothing for the reset to interrupt. Raise REPEAT_COUNT."
    reset_ok=0
fi
for i in $(seq 1 $(( NUM_CONTAINERS - 1 ))); do
    container="${CONTAINER_PREFIX}-${i}"
    if ! pytest_running "$container"; then
        echo ">>> ERROR: ${container}'s workload already finished; it will not span the"
        echo ">>>        reset. Raise REPEAT_COUNT."
        reset_ok=0
    fi
done

if [ "$reset_ok" = "1" ]; then
    # SIGINT, not the default SIGTERM: CPython installs no SIGTERM handler, so
    # pytest would die instantly without fixture teardown, leaving the eth/fabric
    # firmware live on a board we are about to reset.
    echo ">>> Stopping workload in ${container0} (SIGINT)..."
    docker exec "$container0" pkill -INT -f pytest || true

    for _ in $(seq "$STOP_TIMEOUT_SECS"); do
        pytest_running "$container0" || break
        sleep 1
    done
    if pytest_running "$container0"; then
        echo ">>> WARNING: pytest still running after ${STOP_TIMEOUT_SECS}s; sending SIGKILL."
        echo ">>>          The device will NOT have been closed cleanly."
        docker exec "$container0" pkill -KILL -f pytest || true
        sleep 2
    fi
    wait "$container0_initial_pid" || true

    # Reset from the host, not via docker exec: the host tt-smi is newer than the
    # dev image's, and it sees all 32 boards, so device-node targets are
    # unambiguous (in-container, ids are relative to that container's view).
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
        # Without this the confirming run can fail on re-enumeration, not ETH state.
        echo ">>> Reset complete. Settling ${POST_RESET_SETTLE_SECS}s..."
        sleep "$POST_RESET_SETTLE_SECS"
    fi
fi

# --- Step 3: Restart container-0 to confirm the reset devices came back ---
if [ "$reset_ok" = "1" ]; then
    echo ">>> Restarting workload in ${container0}..."
    (
        rc=0
        run_in_container "$container0" 1 "$POST_RESET_RUN_TIMEOUT_SECS" || rc=$?
        echo "$rc" > "${RESULTS_DIR}/${container0}.status"
    ) &
    bg_pids+=($!)
else
    echo 1 > "${RESULTS_DIR}/${container0}.status"
fi

# --- Step 4: Wait for every container's run to finish ---
echo ">>> Waiting for all containers to finish..."
for pid in "${bg_pids[@]}"; do
    wait "$pid" || true
done

echo ">>> All containers finished. Results in ${RESULTS_DIR}/"
exit 0
