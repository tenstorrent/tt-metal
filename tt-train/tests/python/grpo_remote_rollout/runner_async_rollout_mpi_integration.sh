#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ -z "${TT_METAL_HOME:-}" ]]; then
    echo "TT_METAL_HOME is not set" >&2
    exit 1
fi

TEST_DIR="${TT_METAL_HOME}/tt-train/tests/python/grpo_remote_rollout"
EXAMPLE_DIR="${TT_METAL_HOME}/tt-train/sources/examples/grpo_remote_rollout/gsm8k_onestep"
HOST_FILE="${EXAMPLE_DIR}/configurations/split_1_1/hosts.txt"
RANK_BINDINGS_FILE="${EXAMPLE_DIR}/configurations/split_1_1/rank_bindings.yaml"
TEST_FILE="${TEST_DIR}/test_async_rollout_mpi_integration.py"

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
            exit 1 ;;
    esac
    shift
done

# The rank bindings use an MGD path relative to the one-step example directory.
cd "${EXAMPLE_DIR}"

CMD="python3 -m pytest -s -p no:cacheprovider --rootdir=${TEST_DIR} ${TEST_FILE}"

"${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py" \
    --rank-binding "${RANK_BINDINGS_FILE}" \
    --mpi-args "--hostfile ${HOST_FILE} --tag-output --oversubscribe" \
    -- ${CMD}
