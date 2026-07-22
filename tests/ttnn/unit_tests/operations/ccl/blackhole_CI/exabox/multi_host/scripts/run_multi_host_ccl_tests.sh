#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Launches the general multi-host CCL tests in
# tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/multi_host/
# on the BH Galaxy cluster.
#
# Tests covered:
#   - test_all_gather_exabox.py
#   - test_all_reduce_exabox.py
#   - test_reduce_scatter_exabox.py
#   - test_all_broadcast_exabox.py
#   - test_broadcast_exabox.py
#   - test_all_to_all_dispatch_exabox.py
#   - test_all_to_all_combine_exabox.py
#
# Each test file has variants per mesh size (test_*_8x4 / _16x4 / _32x4),
# selected by MESH_DEVICE.

set -uo pipefail

DEFAULT_QUAD_HOSTS="bh-glx-b06u02,bh-glx-b06u08,bh-glx-b07u02,bh-glx-b07u08"
DEFAULT_DUAL_HOSTS="bh-glx-b06u02,bh-glx-b06u08"

usage() {
    cat <<EOF
Usage: $(basename "$0") [-h|--help] [TEST_NODEID ...]

Runs the general multi-host CCL tests against the BH Galaxy cluster.

Arguments:
  TEST_NODEID      Optional. Pytest node ID(s) to run, e.g.
                     test_all_gather_exabox.py::test_all_gather_32x4
                   With no positional args, runs all tests whose name
                   matches the MESH_DEVICE suffix (_8x4 / _16x4 / _32x4)
                   under the multi_host/ folder.

  -h, --help       Show this message and exit.

Required environment:
  TT_METAL_HOME    Repo root. Required — script aborts if unset.
                   Example: export TT_METAL_HOME=/path/to/tt-metal

Optional environment:
  MESH_DEVICE      One of SINGLE_BH | DUAL_BH | QUAD_BH. Default: QUAD_BH.
                     SINGLE_BH (8×4, 1 host)  → runs without tt-run/MPI
                     DUAL_BH   (16×4, 2 hosts) → tt-run, 16x4 binding, _16x4 tests
                     QUAD_BH   (32×4, 4 hosts) → tt-run, 32x4 binding, _32x4 tests

  HOSTS            Comma- or space-separated host list (per MESH_DEVICE).
                   Default for QUAD_BH: ${DEFAULT_QUAD_HOSTS}
                   Default for DUAL_BH: ${DEFAULT_DUAL_HOSTS}
                   *** OVERRIDE THIS for a different cluster. ***
                   Ignored for SINGLE_BH.

  PYTEST_TIMEOUT   Per-test timeout (seconds). Default: 240.

Examples:
  # Default: all QUAD_BH tests on the default cluster
  bash $0

  # DUAL_BH on the user's 2-host subset
  MESH_DEVICE=DUAL_BH HOSTS="hostA,hostB" bash $0

  # SINGLE_BH (no MPI launch)
  MESH_DEVICE=SINGLE_BH bash $0

  # One specific test (file-relative path; mesh suffix must match)
  bash $0 test_all_gather_exabox.py::test_all_gather_32x4

  # Specific parametrized test
  bash $0 'test_all_reduce_exabox.py::test_all_reduce_32x4[small_l1-2links-axis1-32x4_grid-fabric_1d-linear]'

See also:
  RUNBOOK.md (sibling file)            — full prerequisites and troubleshooting.
EOF
}

# --- arg parsing ----------------------------------------------------------

TESTS_TO_RUN=()
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --) shift; while [ $# -gt 0 ]; do TESTS_TO_RUN+=("$1"); shift; done ;;
        -*) echo "[error] unknown flag: $1" >&2; echo "Run with --help for usage." >&2; exit 2 ;;
        *)  TESTS_TO_RUN+=("$1") ;;
    esac
    shift
done

# --- environment ----------------------------------------------------------

if [ -z "${TT_METAL_HOME:-}" ]; then
    echo "[error] TT_METAL_HOME is not set. Export it to your tt-metal repo root, e.g.:" >&2
    echo "          export TT_METAL_HOME=/path/to/tt-metal" >&2
    echo "        Run with --help for usage." >&2
    exit 1
fi

MESH_DEVICE="${MESH_DEVICE:-QUAD_BH}"
PYTEST_TIMEOUT="${PYTEST_TIMEOUT:-240}"
TEST_DIR="tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/multi_host"

case "${MESH_DEVICE}" in
    SINGLE_BH)
        RANK_BINDING=""
        HOSTS=""
        TEST_SUFFIX="_8x4"
        ;;
    DUAL_BH)
        RANK_BINDING="${TT_METAL_HOME}/tests/tt_metal/distributed/config/16x4_dual_bh_galaxy_rank_bindings.yaml"
        HOSTS="${HOSTS:-${DEFAULT_DUAL_HOSTS}}"
        TEST_SUFFIX="_16x4"
        ;;
    QUAD_BH)
        RANK_BINDING="${TT_METAL_HOME}/tests/tt_metal/distributed/config/32x4_quad_bh_galaxy_rank_bindings.yaml"
        HOSTS="${HOSTS:-${DEFAULT_QUAD_HOSTS}}"
        TEST_SUFFIX="_32x4"
        ;;
    *)
        echo "[error] unsupported MESH_DEVICE='${MESH_DEVICE}'. Use SINGLE_BH, DUAL_BH, or QUAD_BH." >&2
        exit 2
        ;;
esac

# Normalize HOSTS: accept space- or comma-separated; mpirun --host wants commas.
if [ -n "${HOSTS}" ]; then
    HOSTS="$(echo "${HOSTS}" | tr ' ' ',' | tr -s ',' | sed 's/^,*//;s/,*$//')"
fi

# Build pytest target list.
if [ "${#TESTS_TO_RUN[@]}" -eq 0 ]; then
    # No test specified: run every test in multi_host/ matching the mesh suffix.
    PYTEST_TARGETS="${TEST_DIR}"
    PYTEST_KEYWORD="-k ${TEST_SUFFIX}"
    TESTS_DESC="(all tests with ${TEST_SUFFIX} in multi_host/)"
else
    # Specific test node IDs: prefix with $TEST_DIR if not already absolute.
    PYTEST_TARGETS=""
    for t in "${TESTS_TO_RUN[@]}"; do
        case "${t}" in
            /*|tests/*) PYTEST_TARGETS="${PYTEST_TARGETS} ${t}" ;;
            *)          PYTEST_TARGETS="${PYTEST_TARGETS} ${TEST_DIR}/${t}" ;;
        esac
    done
    PYTEST_KEYWORD=""   # exact node IDs; no need to filter
    TESTS_DESC="${TESTS_TO_RUN[*]}"
fi

# --- pre-flight banner ----------------------------------------------------

echo "[info] MESH_DEVICE     = ${MESH_DEVICE}"
echo "[info] HOSTS           = ${HOSTS:-(none — single-host run)}"
case "${MESH_DEVICE}" in
    QUAD_BH) [ "${HOSTS}" = "${DEFAULT_QUAD_HOSTS}" ] && echo "[info]                   (default — override with HOSTS=h1,h2,h3,h4 for another cluster)" ;;
    DUAL_BH) [ "${HOSTS}" = "${DEFAULT_DUAL_HOSTS}" ] && echo "[info]                   (default — override with HOSTS=h1,h2 for another cluster)" ;;
esac
[ -n "${RANK_BINDING}" ] && echo "[info] RANK_BINDING    = ${RANK_BINDING}"
echo "[info] PYTEST_TIMEOUT  = ${PYTEST_TIMEOUT}"
echo "[info] tests to run    = ${TESTS_DESC}"
echo

# --- launch ---------------------------------------------------------------

if [ "${MESH_DEVICE}" = "SINGLE_BH" ]; then
    # No tt-run / MPI for single-host runs.
    source "${TT_METAL_HOME}/python_env/bin/activate"
    MESH_DEVICE=SINGLE_BH pytest --timeout=${PYTEST_TIMEOUT} -v ${PYTEST_KEYWORD} ${PYTEST_TARGETS}
else
    tt-run \
        --rank-binding "${RANK_BINDING}" \
        --mpi-args "--host ${HOSTS}" \
        bash -c "source ${TT_METAL_HOME}/python_env/bin/activate && \
                 MESH_DEVICE=${MESH_DEVICE} \
                 pytest --timeout=${PYTEST_TIMEOUT} -v ${PYTEST_KEYWORD} ${PYTEST_TARGETS}"
fi
