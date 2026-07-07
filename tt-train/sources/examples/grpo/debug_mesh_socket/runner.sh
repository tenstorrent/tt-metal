#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Launches the 2-rank MeshSocket 4->4 corruption reproducer via tt-run. Both ranks open a
# single [1, 4] mesh; the receiver opens 4 sockets on that one mesh, socket i carrying the
# single connection (0,i) -> (0,i) ("bigmesh_Nsock"). The sender streams REPRO_NUM_TENSORS
# sharded tensors over all 4 sockets; the receiver issues every recv_async, synchronizes once,
# then verifies -- reproducing the corrupt-tensor bug. See repro_socket_transfer.py.
#
# Plain python script (not pytest) so there is no timeout and the per-step prints stream
# straight to the terminal.
#
# This script does NOT switch git branches -- check out + build the branch you want yourself
# first. The current branch is auto-detected and tagged onto every log line.
#
# Configurable:
#   REPRO_NUM_TENSORS=4 REPRO_TENSOR_SHAPE=1,1,32,32 bash debug_mesh_socket/runner.sh

set -euo pipefail

if [[ -z "${TT_METAL_HOME:-}" ]]; then
    echo "TT_METAL_HOME is not set" >&2
    exit 1
fi

DEBUG_DIR="${TT_METAL_HOME}/tt-train/sources/examples/grpo/debug_mesh_socket"
HOST_FILE="${DEBUG_DIR}/configurations/local8/hosts.txt"
RANK_BINDINGS_FILE="${DEBUG_DIR}/configurations/local8/rank_bindings.yaml"
SCRIPT="${DEBUG_DIR}/repro_socket_transfer.py"

# Forwarded to every rank below; override on the command line to change them.
export REPRO_NUM_TENSORS="${REPRO_NUM_TENSORS:-4}"
export REPRO_TENSOR_SHAPE="${REPRO_TENSOR_SHAPE:-1,1,32,32}"
# Git branch label so every rank's log line says which branch it ran on.
export REPRO_BRANCH="${REPRO_BRANCH:-$(git -C "${DEBUG_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)}"

while [[ "$#" -gt 0 ]]; do
    case "$1" in
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

echo "[runner] branch=${REPRO_BRANCH} mode=bigmesh_Nsock shape=${REPRO_TENSOR_SHAPE} num_tensors=${REPRO_NUM_TENSORS}"

# tt-run resolves the relative `mesh_graph_desc_path` in rank_bindings.yaml against the launch
# directory, so cd into this dir to make "configurations/local8/mgd.textproto" resolve here.
cd "${DEBUG_DIR}"

"${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py" \
    --rank-binding "${RANK_BINDINGS_FILE}" \
    --mpi-args "--hostfile ${HOST_FILE} --tag-output -x REPRO_NUM_TENSORS -x REPRO_TENSOR_SHAPE -x REPRO_BRANCH" \
    python3 "${SCRIPT}"
