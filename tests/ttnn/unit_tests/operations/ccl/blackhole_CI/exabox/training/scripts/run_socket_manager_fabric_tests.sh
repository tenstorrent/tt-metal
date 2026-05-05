#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Launches the SocketManager-FABRIC training tests on QUAD_BH.
#
# These exercise the tt-train tensor-level wrapper:
#     SocketManager.send/recv → MeshSocket → BidirectionalFabricSocket → fabric
# (as opposed to the byte-level DistributedContext.send/recv path covered by
# run_training_tests.sh).
#
# Required artifacts (already in the repo):
#   - tests/tt_metal/distributed/config/quad_bh_galaxy_split_4x2_multi_mesh_rank_bindings.yaml
#       (4 ranks → 4 distinct mesh_ids — MeshSocket precondition)
#   - tests/tt_metal/tt_fabric/custom_mesh_descriptors/quad_bh_galaxy_4mesh_ring_8ch.textproto
#       (4 × (8,4) Galaxy meshes, 8-channel ring inter-mesh edges, matching
#        the physical adjacency on the bh-glx-b06/b07 cluster)
#
# Prereqs (also see RUNBOOK.md §2):
#   - $TT_METAL_HOME built with --build-tt-train.
#   - All 4 hosts in --mpi-args reachable, passwordless ssh.
#   - One tt-run launch at a time. Pytest sometimes hangs in fabric teardown
#     even on success; this script kills lingering processes between tests.
#
# Usage:
#   # Run all 3 FABRIC tests sequentially (default)
#   bash tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/training/scripts/run_socket_manager_fabric_tests.sh
#
#   # Run a single test
#   bash .../run_socket_manager_fabric_tests.sh test_round_robin_send_recv_fabric_32x4
#
#   # Override host list
#   HOSTS="h1,h2,h3,h4" bash .../run_socket_manager_fabric_tests.sh
#
#   # Skip the chip reset (faster for back-to-back runs after a clean exit)
#   SKIP_RESET=1 bash .../run_socket_manager_fabric_tests.sh

set -uo pipefail

DEFAULT_HOSTS="bh-glx-b06u02,bh-glx-b06u08,bh-glx-b07u02,bh-glx-b07u08"

usage() {
    cat <<EOF
Usage: $(basename "$0") [-h|--help] [TEST_NAME ...]

Launches the SocketManager-FABRIC training tests on QUAD_BH (4 ranks).
These exercise the tt-train tensor-level wrapper:
    SocketManager.send/recv → MeshSocket → BidirectionalFabricSocket → fabric

Arguments:
  TEST_NAME        Optional. Run only the named test(s); default is to run
                   all 3 sequentially. Valid names:
                     - test_round_robin_send_recv_fabric_32x4
                     - test_pipeline_activation_handoff_fabric_32x4
                     - test_remote_optimizer_grad_exchange_fabric_32x4

  -h, --help       Show this message and exit.

Required environment:
  TT_METAL_HOME    Repo root. Must be set; build must include --build-tt-train.

Optional environment:
  HOSTS            Comma-separated host list (4 ranks). Default:
                     ${DEFAULT_HOSTS}
                   *** OVERRIDE THIS for a different cluster. ***
                   The default targets the bh-glx-b06/b07 4-host BH Galaxy.
                   Other clusters will need their own hostnames AND likely
                   their own mesh-graph descriptor matching the actual
                   inter-mesh fabric wiring (see RUNBOOK.md §7.5–7.7).

  SKIP_RESET=1     Skip the ~60s chip reset between tests. Use only when
                   you know the previous run exited cleanly. Default: reset.

  LOG_DIR          Where per-test logs go. Default: /tmp/training_send_recv_runs

Examples:
  # Run all 3 FABRIC tests on the default cluster
  bash $0

  # Run a single test
  bash $0 test_round_robin_send_recv_fabric_32x4

  # Run on a different cluster
  HOSTS="my-host-01,my-host-02,my-host-03,my-host-04" bash $0

  # Skip resets (back-to-back run after a clean exit)
  SKIP_RESET=1 bash $0

See also:
  RUNBOOK.md (sibling file) for prerequisites, troubleshooting, and per-test
  manual invocations.
EOF
}

# --- arg parsing ----------------------------------------------------------

TESTS_TO_RUN=()
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            while [ $# -gt 0 ]; do TESTS_TO_RUN+=("$1"); shift; done
            ;;
        -*)
            echo "[error] unknown flag: $1" >&2
            echo "Run with --help for usage." >&2
            exit 2
            ;;
        *)
            TESTS_TO_RUN+=("$1")
            ;;
    esac
    shift
done

# --- environment ----------------------------------------------------------

: "${TT_METAL_HOME:?set TT_METAL_HOME first (run with --help for usage)}"

HOSTS="${HOSTS:-${DEFAULT_HOSTS}}"
SKIP_RESET="${SKIP_RESET:-0}"
RANK_BINDING="${TT_METAL_HOME}/tests/tt_metal/distributed/config/quad_bh_galaxy_split_4x2_multi_mesh_rank_bindings.yaml"
TEST_FILE="tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/training/test_socket_manager_fabric_training.py"
LOG_DIR="${LOG_DIR:-/tmp/training_send_recv_runs}"
mkdir -p "${LOG_DIR}"

ALL_TESTS=(
    test_round_robin_send_recv_fabric_32x4
    test_pipeline_activation_handoff_fabric_32x4
    test_remote_optimizer_grad_exchange_fabric_32x4
)

if [ "${#TESTS_TO_RUN[@]}" -eq 0 ]; then
    TESTS_TO_RUN=("${ALL_TESTS[@]}")
fi

# Show the resolved cluster + reset config so a user running on a different
# cluster has a chance to ctrl-C before chips get touched.
echo "[info] HOSTS         = ${HOSTS}"
if [ "${HOSTS}" = "${DEFAULT_HOSTS}" ]; then
    echo "[info]                 (default — override with HOSTS=h1,h2,h3,h4 if on another cluster)"
fi
echo "[info] RANK_BINDING  = ${RANK_BINDING}"
echo "[info] SKIP_RESET    = ${SKIP_RESET}"
echo "[info] tests to run  = ${TESTS_TO_RUN[*]}"
echo

# IFS-split the comma-separated host list into an array for ssh loops.
IFS=',' read -ra HOST_ARR <<< "${HOSTS}"

reset_chips() {
    echo "[reset] resetting chips on ${HOSTS} (~60s)"
    for h in "${HOST_ARR[@]}"; do
        ssh -o BatchMode=yes "$h" "${TT_METAL_HOME}/python_env/bin/tt-smi -glx_reset_auto" \
            > "${LOG_DIR}/reset_${h}.log" 2>&1 &
    done
    wait
    local fail=0
    for h in "${HOST_ARR[@]}"; do
        if grep -q "Re-initialized 32 boards" "${LOG_DIR}/reset_${h}.log"; then
            echo "[reset] $h: OK"
        else
            echo "[reset] $h: FAIL — see ${LOG_DIR}/reset_${h}.log"
            fail=1
        fi
    done
    return $fail
}

kill_stragglers() {
    # Local
    ps -ef | grep -E 'tt-run|prterun|pytest|prted' | grep -v grep | awk '{print $2}' \
        | xargs -r kill -9 2>/dev/null || true
    # Remote
    for h in "${HOST_ARR[@]}"; do
        ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" \
            "ps -ef | grep -E 'pytest|prted' | grep -v grep | awk '{print \$2}' | xargs -r kill -9" \
            2>/dev/null || true
    done
    sleep 1
}

run_one() {
    local test_name="$1"
    local stamp; stamp=$(date +%Y%m%d_%H%M%S)
    local logfile="${LOG_DIR}/fabric_${stamp}_${test_name}.log"

    echo
    echo "============================================================"
    echo "[run]  ${test_name}"
    echo "[log]  ${logfile}"
    echo "============================================================"

    set +e
    tt-run \
        --rank-binding "${RANK_BINDING}" \
        --mpi-args "--host ${HOSTS}" \
        bash -c "source python_env/bin/activate && \
                 MESH_DEVICE=QUAD_BH pytest --timeout=600 -v \
                 ${TEST_FILE}::${test_name}" \
        > "${logfile}" 2>&1
    local rc=$?
    set -e

    # tt-run sometimes returns non-zero even when the test passed (because
    # pytest hung in teardown and we killed it on next iteration). Read the
    # log to determine the real outcome.
    if grep -q "1 passed in" "${logfile}"; then
        echo "[run]  ${test_name}: PASSED"
        return 0
    fi
    echo "[run]  ${test_name}: FAILED (tt-run rc=${rc}); tail of log:"
    tail -30 "${logfile}" | sed 's/^/[log]    /'
    return 1
}

# ---- main ------------------------------------------------------------

if [ "${SKIP_RESET}" != "1" ]; then
    reset_chips || { echo "[reset] failed; aborting"; exit 1; }
fi

overall_status=0
for test_name in "${TESTS_TO_RUN[@]}"; do
    # Guard each test with a kill_stragglers + reset because pytest tends to
    # leave fabric processes alive after a successful FABRIC test.
    kill_stragglers
    if [ "${SKIP_RESET}" != "1" ]; then
        reset_chips || { echo "[reset] mid-suite reset failed; aborting"; exit 1; }
    fi

    if ! run_one "${test_name}"; then
        overall_status=1
        # Continue to the next test rather than aborting — collecting all
        # failure modes is more useful than stopping at the first one.
    fi
done

# Final cleanup so the cluster is in a known state when the script exits.
kill_stragglers

echo
echo "============================================================"
if [ ${overall_status} -eq 0 ]; then
    echo "[done] all ${#TESTS_TO_RUN[@]} test(s) PASSED"
else
    echo "[done] some tests FAILED — see logs in ${LOG_DIR}/"
fi
echo "============================================================"

exit ${overall_status}
