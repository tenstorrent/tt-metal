#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Launches the 2-rank ThreadedWeightBridge dict-API test via tt-run.
# Rank 0 pushes 5 dicts through the bridge; rank 1 pops via
# `with bridge.receive_weights() as dicts:` and asserts the trail
# matches rank 0's expected summary.

set -euo pipefail

if [[ -z "${TT_METAL_HOME:-}" ]]; then
    echo "TT_METAL_HOME is not set" >&2
    exit 1
fi

FA_DIR="${TT_METAL_HOME}/tt-train/tests/python/grpo_remote_rollout/fully_async"
TESTS_DIR="${TT_METAL_HOME}/tt-train/tests/python/grpo_remote_rollout"
HOST_FILE="${FA_DIR}/configurations/split_1_1/hosts.txt"
RANK_BINDINGS_FILE="${FA_DIR}/configurations/split_1_1/rank_bindings.yaml"
TEST_FILE="${FA_DIR}/test_threaded_weight_bridge.py"

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
            exit 1
            ;;
    esac
    shift
done

# cd here so the relative mesh_graph_desc_path in rank_bindings.yaml resolves
# (tt-run resolves it against cwd, not the bindings file's dir).
cd "${FA_DIR}"

# --rootdir pins pytest's rootdir so the parent conftest.py is picked up.
CMD="python3 -m pytest -s -p no:cacheprovider --rootdir=${TESTS_DIR} ${TEST_FILE}"

"${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py" \
    --rank-binding "${RANK_BINDINGS_FILE}" \
    --mpi-args "--hostfile ${HOST_FILE} --tag-output" \
    -- ${CMD}
