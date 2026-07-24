#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Launches the 2-rank ttml -> tt-transformers WeightBridge test via tt-run.
# Override config locations with --rank-bindings / --hostfile for your setup.

set -euo pipefail

if [[ -z "${TT_METAL_HOME:-}" ]]; then
    echo "TT_METAL_HOME is not set" >&2
    exit 1
fi

WT_DIR="${TT_METAL_HOME}/tt-train/tests/python/grpo_remote_rollout/weight_transfer"
HOST_FILE="${WT_DIR}/configurations/local8/hosts.txt"
RANK_BINDINGS_FILE="${WT_DIR}/configurations/local8/rank_bindings.yaml"
TEST_FILE="${WT_DIR}/test_weight_transfer.py"

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
# (tt-run resolves it against cwd/TT_METAL_HOME, not the bindings file's dir).
cd "${WT_DIR}"

# --rootdir pins pytest's rootdir so the parent conftest.py is picked up.
TESTS_DIR="${TT_METAL_HOME}/tt-train/tests/python/grpo_remote_rollout"
CMD="python3 -m pytest -s -p no:cacheprovider --rootdir=${TESTS_DIR} ${TEST_FILE}"

"${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py" \
    --rank-binding "${RANK_BINDINGS_FILE}" \
    --mpi-args "--hostfile ${HOST_FILE} --tag-output" \
    ${CMD}
