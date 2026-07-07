#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Launches the 2-rank MPISocket transfer reproducer via tt-run. Both ranks open a [1, 4]
# mesh; rank 0 streams REPRO_NUM_TENSORS replicated tensors to rank 1 over a single MPISocket
# (host-staged MPI, addressed by rank -- no fabric send_async/recv_async, no submeshes, no
# per-device connection topology). See repro_socket_transfer.py.
#
# Prerequisite: the ttnn build must expose ttnn._ttnn.multi_device.create_socket / SocketType
# (binding in ttnn/core/distributed/distributed_nanobind.cpp). Rebuild _ttnn if the import fails.
#
# Plain python script (not pytest) so there is no timeout and the per-step prints stream
# straight to the terminal -- the last printed line shows exactly where it stalls if it hangs.
#
# This script does NOT switch git branches -- check out + build the branch you want yourself
# first. The current branch is auto-detected and tagged onto every log line.
#
# Configurable:
#   REPRO_NUM_TENSORS=100 REPRO_TENSOR_SHAPE=1,1,32,32 bash debug_mpi_socket/runner.sh

set -euo pipefail

if [[ -z "${TT_METAL_HOME:-}" ]]; then
    echo "TT_METAL_HOME is not set" >&2
    exit 1
fi

DEBUG_DIR="${TT_METAL_HOME}/tt-train/sources/examples/grpo/debug_mpi_socket"
HOST_FILE="${DEBUG_DIR}/configurations/local8/hosts.txt"
RANK_BINDINGS_FILE="${DEBUG_DIR}/configurations/local8/rank_bindings.yaml"
SCRIPT="${DEBUG_DIR}/repro_socket_transfer.py"

# Forwarded to every rank below; override on the command line to change them.
export REPRO_NUM_TENSORS="${REPRO_NUM_TENSORS:-100}"
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

echo "[runner] branch=${REPRO_BRANCH} transport=MPI shape=${REPRO_TENSOR_SHAPE} num_tensors=${REPRO_NUM_TENSORS}"

# tt-run resolves the relative `mesh_graph_desc_path` in rank_bindings.yaml against the launch
# directory, so cd into this dir to make "configurations/local8/mgd.textproto" resolve here.
cd "${DEBUG_DIR}"

"${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py" \
    --rank-binding "${RANK_BINDINGS_FILE}" \
    --mpi-args "--hostfile ${HOST_FILE} --tag-output -x REPRO_NUM_TENSORS -x REPRO_TENSOR_SHAPE -x REPRO_BRANCH" \
    python3 "${SCRIPT}"
