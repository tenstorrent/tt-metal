#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Launches a GRPO training example on two tt-run ranks with remote rollouts.
# Both ranks run the same script; the ttt_remote_sync config makes rank 1
# serve rollouts.
#
# Usage: run_remote_rollout.sh --example boolq|gsm8k [--split <dir in configurations/>]
#        [--config <yaml>] [--script <py>] [--hostfile <file>] [--rank-bindings <yaml>]

set -euo pipefail

if [[ -z "${TT_METAL_HOME:-}" ]]; then
    echo "TT_METAL_HOME is not set" >&2
    exit 1
fi

GRPO_DIR="${TT_METAL_HOME}/tt-train/sources/examples/grpo"
EXAMPLE=""
SPLIT=""
HOST_FILE=""
RANK_BINDINGS_FILE=""
SCRIPT=""
CONFIG=""

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --example)
            shift; EXAMPLE="$1" ;;
        --split)
            shift; SPLIT="$1" ;;
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

case "${EXAMPLE}" in
    boolq)
        : "${SPLIT:=split_2_2}"
        : "${SCRIPT:=${GRPO_DIR}/boolq/boolq_training_example.py}"
        : "${CONFIG:=tt-train/configs/training_configs/grpo_boolq_llama_1b_remote_rollout.yaml}"
        ;;
    gsm8k)
        : "${SPLIT:=split_1_1}"
        : "${SCRIPT:=${GRPO_DIR}/gsm8k/gsm8k_training_example.py}"
        : "${CONFIG:=tt-train/configs/training_configs/grpo_gsm8k_qwen3_p6b_remote_rollout.yaml}"
        ;;
    *)
        echo "--example must be boolq or gsm8k, got '${EXAMPLE}'" >&2
        exit 1
        ;;
esac

: "${HOST_FILE:=${GRPO_DIR}/configurations/${SPLIT}/hosts.txt}"
: "${RANK_BINDINGS_FILE:=${GRPO_DIR}/configurations/${SPLIT}/rank_bindings.yaml}"

# cd here so the relative mesh_graph_desc_path in rank_bindings.yaml resolves
# (tt-run resolves it against cwd, not the rank_bindings file's directory).
cd "${GRPO_DIR}"

CMD="python3 ${SCRIPT} --config ${CONFIG}"

"${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py" \
    --rank-binding "${RANK_BINDINGS_FILE}" \
    --mpi-args "--hostfile ${HOST_FILE} --tag-output" \
    ${CMD}
