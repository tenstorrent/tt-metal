#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Launches the 2-rank socket-transfer reproducer via tt-run. Both ranks open a
# [1, 4] mesh; the transfer topology is selected by env (see repro_socket_transfer.py):
#   REPRO_USE_SUBMESH / REPRO_SINGLE_SOCKET -> bigmesh_1sock | bigmesh_Nsock | submesh_Nsock
#
# Plain python script (not pytest) so there is no timeout and the per-step prints
# stream straight to the terminal -- the last printed line shows exactly where it
# stalls if it hangs.
#
# This script does NOT switch git branches -- check out + build the branch you want
# (e.g. ichovpan/socket-baseline or ichovpan/socket-patched) yourself first. The
# current branch is auto-detected and tagged onto every log line so it is always
# obvious which branch a log came from.
#
# Configurable:
#   REPRO_NUM_TENSORS=8 REPRO_TENSOR_SHAPE=1,1,8192,4096 \
#   REPRO_USE_SUBMESH=1 REPRO_SINGLE_SOCKET=0 bash mesh_socket_debug/runner.sh

set -euo pipefail

if [[ -z "${TT_METAL_HOME:-}" ]]; then
    echo "TT_METAL_HOME is not set" >&2
    exit 1
fi

DEBUG_DIR="${TT_METAL_HOME}/tt-train/sources/examples/grpo/mesh_socket_debug"
HOST_FILE="${DEBUG_DIR}/configurations/local8/hosts.txt"
RANK_BINDINGS_FILE="${DEBUG_DIR}/configurations/local8/rank_bindings.yaml"
SCRIPT="${DEBUG_DIR}/repro_socket_transfer.py"

# Forwarded to every rank below; override on the command line to change them.
export REPRO_NUM_TENSORS="${REPRO_NUM_TENSORS:-100}"
export REPRO_TENSOR_SHAPE="${REPRO_TENSOR_SHAPE:-1,1,32,32}"
export REPRO_USE_SUBMESH="${REPRO_USE_SUBMESH:-0}"
export REPRO_SINGLE_SOCKET="${REPRO_SINGLE_SOCKET:-0}"
# Git branch label so every rank's log line says which branch (baseline vs patched) it ran on.
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

mode="$([[ "${REPRO_USE_SUBMESH}" = "1" ]] && echo submesh || echo bigmesh)_$([[ "${REPRO_SINGLE_SOCKET}" = "1" ]] && echo 1sock || echo Nsock)"
echo "[runner] branch=${REPRO_BRANCH} mode=${mode} shape=${REPRO_TENSOR_SHAPE} num_tensors=${REPRO_NUM_TENSORS}"

# tt-run resolves the relative `mesh_graph_desc_path` in rank_bindings.yaml
# against the launch directory, so cd into the mesh_socket_debug dir to make
# "configurations/local8/mgd.textproto" resolve here.
cd "${DEBUG_DIR}"

"${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py" \
    --rank-binding "${RANK_BINDINGS_FILE}" \
    --mpi-args "--hostfile ${HOST_FILE} --tag-output -x REPRO_NUM_TENSORS -x REPRO_TENSOR_SHAPE -x REPRO_USE_SUBMESH -x REPRO_SINGLE_SOCKET -x REPRO_BRANCH" \
    python3 "${SCRIPT}"
