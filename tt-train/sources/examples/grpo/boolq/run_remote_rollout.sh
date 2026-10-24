#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Launches the BoolQ GRPO training example on two tt-run ranks with remote
# rollouts (2 TTML chips + 2 TTT chips on one host). Both ranks run the same
# script; the remote_sync config makes rank 1 serve rollouts.

set -euo pipefail

if [[ -z "${TT_METAL_HOME:-}" ]]; then
    echo "TT_METAL_HOME is not set" >&2
    exit 1
fi

EX_DIR="${TT_METAL_HOME}/tt-train/sources/examples/grpo/boolq"
CONFIG_DIR="split_2_2"
HOST_FILE=""
RANK_BINDINGS_FILE=""
SCRIPT="${EX_DIR}/boolq_training_example.py"
CONFIG="tt-train/configs/training_configs/grpo_boolq_llama_1b_remote_rollout.yaml"

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --hostfile)
            shift; HOST_FILE="$1" ;;
        --rank-bindings)
            shift; RANK_BINDINGS_FILE="$1" ;;
        --script)
            shift; SCRIPT="$1" ;;
        --config)
            shift; CONFIG="$1" ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
    shift
done

: "${HOST_FILE:=${EX_DIR}/configurations/${CONFIG_DIR}/hosts.txt}"
: "${RANK_BINDINGS_FILE:=${EX_DIR}/configurations/${CONFIG_DIR}/rank_bindings.yaml}"

# cd here so the relative mesh_graph_desc_path in rank_bindings.yaml resolves
# (tt-run resolves it against cwd, not the rank_bindings file's directory).
cd "${EX_DIR}"

CMD="python3 ${SCRIPT} --config ${CONFIG}"

"${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py" \
    --rank-binding "${RANK_BINDINGS_FILE}" \
    --mpi-args "--hostfile ${HOST_FILE} --tag-output" \
    ${CMD}
