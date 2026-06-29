#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Launches the 2-rank DDP-TTML -> per-submesh-TTT weight transfer test via
# tt-run. Rank 0 opens a [1, 4] DDP ttml mesh; rank 1 opens a [1, 4] parent
# mesh and splits it into four [1, 1] submeshes (8 chips total).
#
# tt-run wraps mpirun. It takes:
#   --rank-binding   maps each MPI rank to a (mesh_id, mesh_host_rank) and
#                    sets per-rank env (e.g. TT_VISIBLE_DEVICES).
#   --mpi-args       passed straight to mpirun. We use --hostfile here so
#                    both ranks land on localhost.
#
# The configurations/local8/* files are a HARDWARE-SPECIFIC TEMPLATE -- see
# their headers. Override config locations with --rank-bindings / --hostfile
# (or edit local8/*) to match your machine.

set -euo pipefail

if [[ -z "${TT_METAL_HOME:-}" ]]; then
    echo "TT_METAL_HOME is not set" >&2
    exit 1
fi

WT_DIR="${TT_METAL_HOME}/tt-train/sources/examples/grpo/tests/weight_transfer"
# 4-devices-per-side config: each rank opens its full [1, 4] mesh -- rank 0 as
# the DDP ttml mesh, rank 1 split into four [1, 1] submeshes (8 chips total).
# The configurations/local8/* files are a HARDWARE-SPECIFIC TEMPLATE (see their
# headers); override with --rank-bindings / --hostfile to match your machine.
HOST_FILE="${WT_DIR}/configurations/local8/hosts.txt"
RANK_BINDINGS_FILE="${WT_DIR}/configurations/local8/rank_bindings.yaml"
TEST_FILE="${WT_DIR}/test_ddp_submesh_transfer.py"

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

# tt-run resolves the relative `mesh_graph_desc_path` inside
# rank_bindings.yaml against (1) TT_METAL_HOME, (2) the launch directory
# (cwd at tt-run invocation), and (3) cwd at resolution time -- not against
# the rank_bindings file's own directory. cd into the weight_transfer dir so
# "configurations/local8/mgd.textproto" resolves here.
cd "${WT_DIR}"

# `pytest -s` keeps the rank-tagged prints visible; -p no:cacheprovider
# avoids a stale .pytest_cache being shared across ranks. `--rootdir` pins
# pytest's rootdir to the grpo tests dir so the parent conftest.py (sys.path
# setup, fabric config) is picked up the same way as a non-tt-run run.
TESTS_DIR="${TT_METAL_HOME}/tt-train/sources/examples/grpo/tests"
CMD="python3 -m pytest -s -p no:cacheprovider --rootdir=${TESTS_DIR} ${TEST_FILE}"

"${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py" \
    --rank-binding "${RANK_BINDINGS_FILE}" \
    --mpi-args "--hostfile ${HOST_FILE} --tag-output" \
    ${CMD}
