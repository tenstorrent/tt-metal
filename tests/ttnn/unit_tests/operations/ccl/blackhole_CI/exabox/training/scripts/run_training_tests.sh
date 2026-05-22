#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Launches the byte-level DistributedContext.send/recv exabox tests on QUAD_BH.
#
# These exercise the host-MPI primitive directly via Python bindings on
# ttml.core.distributed.DistributedContext (see
# tt-train/sources/ttml/nanobind/nb_core.cpp).
#
# Run with --help for full usage.

set -uo pipefail

DEFAULT_HOSTS="bh-glx-b06u02,bh-glx-b06u08,bh-glx-b07u02,bh-glx-b07u08"

usage() {
    cat <<EOF
Usage: $(basename "$0") [-h|--help] [TEST_NAME ...]

Launches the byte-level DistributedContext.send/recv tests on QUAD_BH.
By default runs all 3 tests in one tt-run / pytest invocation.

Arguments:
  TEST_NAME        Optional. Run only the named test(s); default is all 3.
                   Valid names:
                     - test_round_robin_send_recv_32x4
                     - test_pipeline_activation_handoff_32x4
                     - test_remote_optimizer_grad_exchange_32x4

  -h, --help       Show this message and exit.

Required environment:
  TT_METAL_HOME    Repo root. Must be set; build must include --build-tt-train.

Optional environment:
  HOSTS            Host list (4 ranks); comma- or space-separated (mixed
                   also accepted, e.g. "h1,h2 h3 h4"). Default:
                     ${DEFAULT_HOSTS}
                   *** OVERRIDE THIS for a different cluster. ***
                   The default targets the bh-glx-b06/b07 4-host BH Galaxy.
                   Byte-level tests are topology-agnostic (pure host MPI),
                   so any 4-host BH Galaxy cluster should work as long as
                   ssh between hosts is configured.

Examples:
  # All 3 tests on the default cluster
  bash $0

  # Run a single test
  bash $0 test_pipeline_activation_handoff_32x4

  # Run two specific tests
  bash $0 test_round_robin_send_recv_32x4 test_remote_optimizer_grad_exchange_32x4

  # Different cluster — commas, spaces, or a mix all work
  HOSTS="my-host-01,my-host-02,my-host-03,my-host-04" bash $0
  HOSTS="my-host-01 my-host-02 my-host-03 my-host-04" bash $0

See also:
  RUNBOOK.md (sibling file) — full prerequisites and troubleshooting.
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

# Normalize HOSTS to comma-separated for mpirun's --host flag.
# Accept commas, spaces, tabs, or any mix; collapse runs and trim edges.
HOSTS="${HOSTS//$'\t'/ }"
HOSTS="${HOSTS// /,}"
while [[ "${HOSTS}" == *,,* ]]; do HOSTS="${HOSTS//,,/,}"; done
HOSTS="${HOSTS#,}"
HOSTS="${HOSTS%,}"

NUM_HOSTS="$(awk -F, '{print NF}' <<< "${HOSTS}")"
if [ "${NUM_HOSTS}" -ne 4 ]; then
    echo "[error] HOSTS must contain exactly 4 hosts (got ${NUM_HOSTS}): ${HOSTS}" >&2
    echo "Run with --help for usage." >&2
    exit 2
fi

RANK_BINDING="${TT_METAL_HOME}/tests/tt_metal/distributed/config/32x4_quad_bh_galaxy_rank_bindings.yaml"
TEST_FILE="tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/training/test_send_recv_training.py"

# Build the pytest target list. With no positional args, run the whole file
# (all 3 tests). With one or more TEST_NAMEs, run those specific node IDs.
if [ "${#TESTS_TO_RUN[@]}" -eq 0 ]; then
    PYTEST_TARGETS="${TEST_FILE}"
    TESTS_DESC="(all 3 tests in ${TEST_FILE##*/})"
else
    PYTEST_TARGETS=""
    for t in "${TESTS_TO_RUN[@]}"; do
        PYTEST_TARGETS="${PYTEST_TARGETS} ${TEST_FILE}::${t}"
    done
    TESTS_DESC="${TESTS_TO_RUN[*]}"
fi

# Show the resolved cluster so a user running on a different one has a chance
# to ctrl-C before tt-run starts touching chips.
echo "[info] HOSTS         = ${HOSTS}"
if [ "${HOSTS}" = "${DEFAULT_HOSTS}" ]; then
    echo "[info]                 (default — override with HOSTS=h1,h2,h3,h4 if on another cluster)"
fi
echo "[info] RANK_BINDING  = ${RANK_BINDING}"
echo "[info] tests to run  = ${TESTS_DESC}"
echo

tt-run \
    --rank-binding "${RANK_BINDING}" \
    --mpi-args "--host ${HOSTS}" \
    bash -c "source python_env/bin/activate && \
             MESH_DEVICE=QUAD_BH \
             pytest --timeout=300 -v ${PYTEST_TARGETS}"
