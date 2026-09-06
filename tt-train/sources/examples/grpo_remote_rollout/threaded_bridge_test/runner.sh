#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Launches the 2-rank threaded_bridge_test via tt-run. Rank 0 pushes a
# handful of tensors through ThreadedWeightBridge; rank 1 verifies and
# prints [PASS] / [FAIL].
#
# Configuration is a verbatim copy of gsm8k_fully_async/configurations/split_1_1
# (two [1, 1] Blackhole meshes, one no-op fabric intermesh connection).

set -euo pipefail

if [[ -z "${TT_METAL_HOME:-}" ]]; then
    echo "TT_METAL_HOME is not set" >&2
    exit 1
fi

EX_DIR="${TT_METAL_HOME}/tt-train/sources/examples/grpo_remote_rollout/threaded_bridge_test"
CONFIG_DIR="split_1_1"
HOST_FILE=""
RANK_BINDINGS_FILE=""
SCRIPT="${EX_DIR}/test_threaded_bridge.py"

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

: "${HOST_FILE:=${EX_DIR}/configurations/${CONFIG_DIR}/hosts.txt}"
: "${RANK_BINDINGS_FILE:=${EX_DIR}/configurations/${CONFIG_DIR}/rank_bindings.yaml}"

# cd here so the relative mesh_graph_desc_path in rank_bindings.yaml resolves
# (tt-run resolves it against cwd, not the rank_bindings file's directory).
cd "${EX_DIR}"

CMD="python3 ${SCRIPT}"

"${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py" \
    --rank-binding "${RANK_BINDINGS_FILE}" \
    --mpi-args "--hostfile ${HOST_FILE} --tag-output" \
    ${CMD}
