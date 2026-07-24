#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Launches the 2-rank GRPO BoolQ training example via tt-run.
# --topology: 2x2 (default, configurations/local4) or 4x4 (local8).
# NOTE: 4x4 currently hangs somewhere in the cross-rank handshake.

set -euo pipefail

if [[ -z "${TT_METAL_HOME:-}" ]]; then
    echo "TT_METAL_HOME is not set" >&2
    exit 1
fi

EX_DIR="${TT_METAL_HOME}/tt-train/sources/examples/grpo_remote_rollout/boolq"
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

# Map the topology to its configurations/<dir> (explicit flags still win).
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

# cd here so the relative mesh_graph_desc_path in rank_bindings.yaml resolves
# (tt-run resolves it against cwd, not the rank_bindings file's directory).
cd "${EX_DIR}"

CMD="python3 ${SCRIPT}"

"${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py" \
    --rank-binding "${RANK_BINDINGS_FILE}" \
    --mpi-args "--hostfile ${HOST_FILE} --tag-output -x GRPO_BOOLQ_TOPOLOGY" \
    ${CMD}
