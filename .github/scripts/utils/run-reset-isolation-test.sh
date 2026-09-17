#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# run-reset-isolation-test.sh
#
# Every container but container-0 holds one long run spanning the reset; a device
# that was never closed cannot be reset, so container-0 finishes a short run first.
#
# Usage:
#   bash run-reset-isolation-test.sh <NUM_CONTAINERS> <CONTAINER_PREFIX> \
#       <TEST_PATH> <TEST_ARGS> <RESET_DEVICE_IDS> <REPEAT_COUNT>
#
# Always exits 0; per-container .status files carry the result.

set -euo pipefail

NUM_CONTAINERS="${1:?NUM_CONTAINERS required}"
CONTAINER_PREFIX="${2:?CONTAINER_PREFIX required}"
TEST_PATH="${3:?TEST_PATH required}"
TEST_ARGS="${4}"
RESET_DEVICE_IDS="${5:?RESET_DEVICE_IDS required}"
# Repetitions inside one pytest process, sized so the survivor runs outlast the reset.
REPEAT_COUNT="${6:?REPEAT_COUNT required}"

# Matches ResetUtil.post_reset_settle_seconds in tests/sweep_framework/framework/tt_smi_util.py.
POST_RESET_SETTLE_SECS=10
# Bound on container-0's runs so a device that never comes back cannot burn the
# whole step budget and lose the survivors' results.
C0_RUN_TIMEOUT_SECS=900
# The host's tt-smi is a venv console script, not on PATH. Its shebang points at
# the venv python, so the absolute path needs no activation.
HOST_TT_SMI=/opt/tt_metal_infra/provisioning/provisioning_env/bin/tt-smi

RESULTS_DIR=".multi-user-test-results"
mkdir -p "$RESULTS_DIR"
rm -f "${RESULTS_DIR}"/*.status

container0="${CONTAINER_PREFIX}-0"
# Survivors whose run ended before the reset finished; their status is forced to
# a failure at the end, since a pass would only mean "ran, then the reset happened".
declare -a not_spanned=()

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

# --- Step 2: Let container-0 finish a run, so its devices are closed cleanly ---
# Not interrupted with a signal: SIGINT is the only one that tears pytest down
# cleanly, and it did not reach the CCL suite at all in run 34928391619.
echo ">>> Running a short workload in ${container0} to completion..."
reset_ok=1
if ! run_in_container "$container0" 1 "$C0_RUN_TIMEOUT_SECS"; then
    echo ">>> ERROR: ${container0}'s pre-reset workload failed; not resetting."
    reset_ok=0
fi

if [ "$reset_ok" = "1" ]; then
    # From the host: its tt-smi is newer than the dev image's and sees all 32
    # boards, so device-node targets mean what the tray mapping says.
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

        # A survivor that finished before the reset completed never had a
        # workload running across it, so its exit code proves nothing.
        for i in $(seq 1 $(( NUM_CONTAINERS - 1 ))); do
            container="${CONTAINER_PREFIX}-${i}"
            if ! pytest_running "$container"; then
                echo ">>> ERROR: ${container}'s workload ended before the reset completed;"
                echo ">>>        it did not span the reset. Raise REPEAT_COUNT."
                not_spanned+=("$container")
            fi
        done
    fi
fi

# --- Step 3: Run in container-0 again to confirm the reset devices came back ---
if [ "$reset_ok" = "1" ]; then
    echo ">>> Re-running workload in ${container0}..."
    (
        rc=0
        run_in_container "$container0" 1 "$C0_RUN_TIMEOUT_SECS" || rc=$?
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

if [ ${#not_spanned[@]} -gt 0 ]; then
    for container in "${not_spanned[@]}"; do
        echo 1 > "${RESULTS_DIR}/${container}.status"
        echo ">>> ${container}: recorded as failed (workload did not span the reset)."
    done
fi

echo ">>> All containers finished. Results in ${RESULTS_DIR}/"
exit 0
