#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Launches the 2-rank MeshSocket (fabric) 2->6 reproducer via tt-run. Sender opens a [1, 2]
# mesh (1 board), receiver a [1, 6] mesh (3 boards); a single connection (0,0) -> (0,0) carries
# device 0's shard, then the receiver broadcasts it across its [1, 6].
#
# !!! EXPECTED TO FATAL AT open_mesh_device !!! The asymmetric single-board inter-mesh split
# trips the T3000 fabric routing limit ("one src to multiple dst chips ... not supported yet")
# during control-plane bring-up, before any socket. The working symmetric version is
# debug_mesh_socket_and_broadcast (4/4). See repro_socket_transfer.py.
#
# Plain python script (not pytest) so there is no timeout and the per-step prints stream
# straight to the terminal -- the last printed line shows exactly where it stalls if it hangs.
#
# This script does NOT switch git branches -- check out + build the branch you want yourself
# first. The current branch is auto-detected and tagged onto every log line.
#
# Configurable:
#   REPRO_NUM_TENSORS=100 REPRO_TENSOR_SHAPE=1,1,32,32 bash debug_mesh_socket_2_6/runner.sh

set -euo pipefail

if [[ -z "${TT_METAL_HOME:-}" ]]; then
    echo "TT_METAL_HOME is not set" >&2
    exit 1
fi

DEBUG_DIR="${TT_METAL_HOME}/tt-train/sources/examples/grpo/debug_mesh_socket_2_6"
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

echo "[runner] branch=${REPRO_BRANCH} transport=MeshSocket(fabric) split=2->6 shape=${REPRO_TENSOR_SHAPE} num_tensors=${REPRO_NUM_TENSORS}"

# tt-run resolves the relative `mesh_graph_desc_path` in rank_bindings.yaml against the launch
# directory, so cd into this dir to make "configurations/local8/mgd.textproto" resolve here.
cd "${DEBUG_DIR}"

"${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py" \
    --rank-binding "${RANK_BINDINGS_FILE}" \
    --mpi-args "--hostfile ${HOST_FILE} --tag-output -x REPRO_NUM_TENSORS -x REPRO_TENSOR_SHAPE -x REPRO_BRANCH" \
    python3 "${SCRIPT}"
