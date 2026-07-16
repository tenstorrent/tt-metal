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
Usage: $(basename "$0") [OPTIONS] [TEST_NODEID ...]

Runs the general multi-host CCL tests against the BH Galaxy cluster.

Arguments:
  TEST_NODEID      Optional. Pytest node ID(s) to run, e.g.
                     test_all_gather_exabox.py::test_all_gather_32x4
                   With no positional args, runs all tests whose name
                   matches the MESH_DEVICE suffix (_8x4 / _16x4 / _32x4)
                   under the multi_host/ folder.

Options:
  --image <image|none>  Docker image for running tests (default: none).
                        "none" uses local tt-run + python_env (bare-metal).
                        Any other value uses mpi-docker with that image.
  --mpi-if <iface>      Network interface for MPI TCP transport when using
                        Docker mode (passed to mpi-docker --mpi-interface).
                        Ignored in bare-metal mode (tt-run handles MPI).
  -h, --help            Show this message and exit.

Required environment:
  TT_METAL_HOME    Repo root. Required — script aborts if unset.
                   Example: export TT_METAL_HOME=/path/to/tt-metal

Optional environment:
  MESH_DEVICE      One of SINGLE_BH | DUAL_BH | QUAD_BH. Default: QUAD_BH.
                     SINGLE_BH (8×4, 1 host)  → runs without MPI (bare-metal)
                                                 or single-host docker run
                     DUAL_BH   (16×4, 2 hosts) → tt-run or mpi-docker
                     QUAD_BH   (32×4, 4 hosts) → tt-run or mpi-docker

  HOSTS            Comma- or space-separated host list (per MESH_DEVICE).
                   Default for QUAD_BH: ${DEFAULT_QUAD_HOSTS}
                   Default for DUAL_BH: ${DEFAULT_DUAL_HOSTS}
                   *** OVERRIDE THIS for a different cluster. ***
                   Ignored for SINGLE_BH.

  DOCKER_IMAGE     Alternative to --image flag. Flag takes precedence.

  MPI_IF           Alternative to --mpi-if flag. Flag takes precedence.

  PYTEST_TIMEOUT   Per-test timeout (seconds). Default: 240.

Examples:
  # Default: all QUAD_BH tests, bare-metal (no Docker)
  bash $0

  # QUAD_BH with Docker image
  bash $0 --image ghcr.io/tenstorrent/tt-metal/upstream-tests-bh-glx:latest

  # DUAL_BH on the user's 2-host subset with Docker
  MESH_DEVICE=DUAL_BH HOSTS="hostA,hostB" bash $0 --image <image>

  # SINGLE_BH with Docker (single host, no MPI)
  MESH_DEVICE=SINGLE_BH bash $0 --image <image>

  # Explicit bare-metal (same as default)
  bash $0 --image none

  # One specific test (file-relative path; mesh suffix must match)
  bash $0 test_all_gather_exabox.py::test_all_gather_32x4

See also:
  RUNBOOK.md (sibling file)            — full prerequisites and troubleshooting.
EOF
}

# --- arg parsing ----------------------------------------------------------

DOCKER_IMAGE="${DOCKER_IMAGE:-none}"
MPI_IF="${MPI_IF:-}"
TESTS_TO_RUN=()

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --image)
            if [ -z "${2:-}" ]; then
                echo "[error] --image requires a value" >&2; exit 2
            fi
            DOCKER_IMAGE="$2"; shift 2 ;;
        --image=*) DOCKER_IMAGE="${1#--image=}"; shift ;;
        --mpi-if)
            if [ -z "${2:-}" ]; then
                echo "[error] --mpi-if requires a value" >&2; exit 2
            fi
            MPI_IF="$2"; shift 2 ;;
        --mpi-if=*) MPI_IF="${1#--mpi-if=}"; shift ;;
        --) shift; while [ $# -gt 0 ]; do TESTS_TO_RUN+=("$1"); shift; done ;;
        -*) echo "[error] unknown flag: $1" >&2; echo "Run with --help for usage." >&2; exit 2 ;;
        *)  TESTS_TO_RUN+=("$1"); shift ;;
    esac
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

RANK_BINDING=""
MESH_GRAPH_DESC=""

case "${MESH_DEVICE}" in
    SINGLE_BH)
        HOSTS=""
        TEST_SUFFIX="_8x4"
        ;;
    DUAL_BH)
        RANK_BINDING="${TT_METAL_HOME}/tests/tt_metal/distributed/config/16x4_dual_bh_galaxy_rank_bindings.yaml"
        MESH_GRAPH_DESC="tt_metal/fabric/mesh_graph_descriptors/16x4_dual_bh_galaxy_2d_mesh_graph_descriptor.textproto"
        HOSTS="${HOSTS:-${DEFAULT_DUAL_HOSTS}}"
        TEST_SUFFIX="_16x4"
        ;;
    QUAD_BH)
        RANK_BINDING="${TT_METAL_HOME}/tests/tt_metal/distributed/config/32x4_quad_bh_galaxy_rank_bindings.yaml"
        MESH_GRAPH_DESC="tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto"
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
    PYTEST_TARGETS="${TEST_DIR}"
    PYTEST_KEYWORD="-k ${TEST_SUFFIX}"
    TESTS_DESC="(all tests with ${TEST_SUFFIX} in multi_host/)"
else
    PYTEST_TARGETS=""
    for t in "${TESTS_TO_RUN[@]}"; do
        case "${t}" in
            /*|tests/*) PYTEST_TARGETS="${PYTEST_TARGETS} ${t}" ;;
            *)          PYTEST_TARGETS="${PYTEST_TARGETS} ${TEST_DIR}/${t}" ;;
        esac
    done
    PYTEST_KEYWORD=""
    TESTS_DESC="${TESTS_TO_RUN[*]}"
fi

# --- pre-flight banner ----------------------------------------------------

echo "[info] MESH_DEVICE     = ${MESH_DEVICE}"
echo "[info] HOSTS           = ${HOSTS:-(none — single-host run)}"
case "${MESH_DEVICE}" in
    QUAD_BH) [ "${HOSTS}" = "${DEFAULT_QUAD_HOSTS}" ] && echo "[info]                   (default — override with HOSTS=h1,h2,h3,h4 for another cluster)" ;;
    DUAL_BH) [ "${HOSTS}" = "${DEFAULT_DUAL_HOSTS}" ] && echo "[info]                   (default — override with HOSTS=h1,h2 for another cluster)" ;;
esac
echo "[info] DOCKER_IMAGE    = ${DOCKER_IMAGE}"
[ -n "${RANK_BINDING}" ] && echo "[info] RANK_BINDING    = ${RANK_BINDING}"
[ -n "${MESH_GRAPH_DESC}" ] && echo "[info] MESH_GRAPH_DESC = ${MESH_GRAPH_DESC}"
[ -n "${MPI_IF}" ] && echo "[info] MPI_IF          = ${MPI_IF}"
echo "[info] PYTEST_TIMEOUT  = ${PYTEST_TIMEOUT}"
echo "[info] tests to run    = ${TESTS_DESC}"
echo

# --- launch ---------------------------------------------------------------

MPI_DOCKER="${TT_METAL_HOME}/tools/scaleout/exabox/mpi-docker"

if [ "${DOCKER_IMAGE}" = "none" ]; then
    # ====== Bare-metal path (original behavior) ======
    if [ "${MESH_DEVICE}" = "SINGLE_BH" ]; then
        source "${TT_METAL_HOME}/python_env/bin/activate"
        MESH_DEVICE=SINGLE_BH pytest --timeout="${PYTEST_TIMEOUT}" -v ${PYTEST_KEYWORD} ${PYTEST_TARGETS}
    else
        tt-run \
            --rank-binding "${RANK_BINDING}" \
            --mpi-args "--host ${HOSTS}" \
            bash -c "source ${TT_METAL_HOME}/python_env/bin/activate && \
                     MESH_DEVICE=${MESH_DEVICE} \
                     pytest --timeout=${PYTEST_TIMEOUT} -v ${PYTEST_KEYWORD} ${PYTEST_TARGETS}"
    fi
else
    # ====== Docker path (mpi-docker on host) ======
    if [ ! -x "${MPI_DOCKER}" ]; then
        echo "[error] mpi-docker not found at ${MPI_DOCKER}" >&2
        echo "        Ensure TT_METAL_HOME points to a tt-metal checkout." >&2
        exit 1
    fi

    MPI_IF_ARGS=()
    if [ -n "${MPI_IF}" ]; then
        MPI_IF_ARGS=(--mpi-interface "${MPI_IF}")
    fi

    PYTEST_CMD="pytest --timeout=${PYTEST_TIMEOUT} -v ${PYTEST_KEYWORD} ${PYTEST_TARGETS}"

    if [ "${MESH_DEVICE}" = "SINGLE_BH" ]; then
        # Single-host Docker: direct docker run (no MPI needed).
        docker run \
            --rm --net=host --privileged \
            --device /dev/tenstorrent \
            -v /dev/hugepages:/dev/hugepages \
            -v /dev/hugepages-1G:/dev/hugepages-1G \
            -v /etc/udev/rules.d:/etc/udev/rules.d \
            --entrypoint="" \
            "${DOCKER_IMAGE}" \
            bash -c "cd \"\$TT_METAL_HOME\" && MESH_DEVICE=SINGLE_BH ${PYTEST_CMD}"
    else
        # Multi-host Docker: mpi-docker dispatches one container per host.
        "${MPI_DOCKER}" \
            --image "${DOCKER_IMAGE}" \
            --empty-entrypoint \
            "${MPI_IF_ARGS[@]}" \
            --host "${HOSTS}" \
            -x TT_MESH_ID=0 \
            -x TT_MESH_GRAPH_DESC_PATH="${MESH_GRAPH_DESC}" \
            -x MESH_DEVICE="${MESH_DEVICE}" \
            bash -c "cd \"\$TT_METAL_HOME\" && ${PYTEST_CMD}"
    fi
fi
