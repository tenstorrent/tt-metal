#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Launches the 2-rank GRPO BoolQ training example via tt-run.
#
# Topology (--topology, default 2x2; mirrors tests/weight_transfer/runner.sh):
#   2x2 -> configurations/local4 (4 chips total)
#     Rank 0 (TTML): owns one N300 board, opens a [1, 2] DDP mesh, drives
#                    training via ttml + GRPOTrainer.
#     Rank 1 (TTT):  owns one N300 board, opens a [1, 2] parent mesh split
#                    into two [1, 1] submeshes, one tt-transformers
#                    generation worker per submesh, served over RPC.
#   4x4 -> configurations/local8 (8 chips total)
#     Same as above but [1, 4] meshes / four submeshes (two boards per rank).
#     NOTE: 4x4 currently hangs somewhere in the cross-rank handshake.
#
# The selected --topology is exported as GRPO_BOOLQ_TOPOLOGY (and forwarded to
# both ranks via mpirun -x) so boolq_training_example.py opens the matching
# ttml DDP mesh / ttt submeshes.
#
# tt-run wraps mpirun:
#   --rank-binding   maps each MPI rank to a (mesh_id, mesh_host_rank)
#                    and sets per-rank env (e.g. TT_VISIBLE_DEVICES).
#   --mpi-args       passed straight to mpirun. We use --hostfile here
#                    so both ranks land on localhost.
#
# Override config locations with --rank-bindings / --hostfile if you
# adapt this to a multi-host or larger-mesh setup.

set -euo pipefail

if [[ -z "${TT_METAL_HOME:-}" ]]; then
    echo "TT_METAL_HOME is not set" >&2
    exit 1
fi

EX_DIR="${TT_METAL_HOME}/tt-train/sources/examples/grpo/boolq"
TOPOLOGY="2x2"
HOST_FILE=""
RANK_BINDINGS_FILE=""
SCRIPT="${EX_DIR}/boolq_training_example.py"

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --topology)
            shift; TOPOLOGY="$1" ;;
        --hostfile)
            shift; HOST_FILE="$1" ;;
        --rank-bindings)
            shift; RANK_BINDINGS_FILE="$1" ;;
        --script)
            shift; SCRIPT="$1" ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
    shift
done

# Map the topology to its configurations/<dir>. --hostfile / --rank-bindings
# still win if the caller passes them explicitly.
case "${TOPOLOGY}" in
    2x2) CONFIG_DIR="local4" ;;
    4x4) CONFIG_DIR="local8" ;;
    *)
        echo "Unknown --topology: ${TOPOLOGY} (expected 2x2 or 4x4)" >&2
        exit 1
        ;;
esac

: "${HOST_FILE:=${EX_DIR}/configurations/${CONFIG_DIR}/hosts.txt}"
: "${RANK_BINDINGS_FILE:=${EX_DIR}/configurations/${CONFIG_DIR}/rank_bindings.yaml}"

# Both ranks read this to open the matching mesh / submesh count.
export GRPO_BOOLQ_TOPOLOGY="${TOPOLOGY}"

# tt-run resolves the relative `mesh_graph_desc_path` inside
# rank_bindings.yaml against (1) TT_METAL_HOME, (2) the launch
# directory (cwd at tt-run invocation), and (3) cwd at resolution
# time -- not against the rank_bindings file's own directory. cd into
# the example dir so "configurations/<dir>/mgd.textproto" resolves
# here.
cd "${EX_DIR}"

CMD="python3 ${SCRIPT}"

"${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py" \
    --rank-binding "${RANK_BINDINGS_FILE}" \
    --mpi-args "--hostfile ${HOST_FILE} --tag-output -x GRPO_BOOLQ_TOPOLOGY" \
    ${CMD}
