#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Launches the 2-rank HostWeightBridge transport unit test via tt-run.
# Shape 4->4: 4 chips per rank (8 total, the configurations/4_4 topology):
#   rank 0 opens a [1, 4] sender mesh; rank 1 opens a [1, 4] parent and splits
#   it into four [1, 1] submeshes.
#
# tt-run wraps mpirun. --rank-binding maps each MPI rank to a (mesh_id,
# mesh_host_rank) and sets per-rank env (e.g. TT_VISIBLE_DEVICES); --mpi-args is
# passed straight to mpirun. Override config locations with --rank-bindings /
# --hostfile to match your machine.

set -euo pipefail

if [[ -z "${TT_METAL_HOME:-}" ]]; then
    echo "TT_METAL_HOME is not set" >&2
    exit 1
fi

WB_DIR="${TT_METAL_HOME}/tt-train/sources/examples/grpo/tests/weight_bridge"
TESTS_DIR="${TT_METAL_HOME}/tt-train/sources/examples/grpo/tests"
HOST_FILE="${WB_DIR}/configurations/4_4/hosts.txt"
RANK_BINDINGS_FILE="${WB_DIR}/configurations/4_4/rank_bindings.yaml"
TEST_FILE="${WB_DIR}/test_weight_bridge.py"

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --hostfile)
            shift; HOST_FILE="$1" ;;
        --rank-bindings)
            shift; RANK_BINDINGS_FILE="$1" ;;
        --test-file)
            shift; TEST_FILE="$1" ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
    shift
done

# tt-run resolves the relative `mesh_graph_desc_path` inside rank_bindings.yaml
# against (1) TT_METAL_HOME, (2) the launch directory, and (3) cwd -- not against
# the rank_bindings file's own directory. cd into the weight_bridge dir so
# "configurations/4_4/mgd.textproto" resolves here.
cd "${WB_DIR}"

# `pytest -s` keeps the rank-tagged prints visible; -p no:cacheprovider avoids a
# stale .pytest_cache shared across ranks. `--rootdir` pins pytest's rootdir to
# the grpo tests dir so the parent conftest.py (sys.path setup, fabric config)
# is picked up the same way as a non-tt-run run.
CMD="python3 -m pytest -s -p no:cacheprovider --rootdir=${TESTS_DIR} ${TEST_FILE}"

"${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py" \
    --rank-binding "${RANK_BINDINGS_FILE}" \
    --mpi-args "--hostfile ${HOST_FILE} --tag-output" \
    ${CMD}
