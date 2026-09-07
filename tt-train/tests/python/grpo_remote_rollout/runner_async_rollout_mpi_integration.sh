#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ -z "${TT_METAL_HOME:-}" ]]; then
    echo "TT_METAL_HOME is not set" >&2
    exit 1
fi

TEST_DIR="${TT_METAL_HOME}/tt-train/tests/python/grpo_remote_rollout"
CONFIG_DIR="${TEST_DIR}/configurations/independent_1x1"
HOST_FILE="${CONFIG_DIR}/hosts.txt"
RANK_BINDINGS_FILE="${CONFIG_DIR}/rank_bindings.yaml"
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

# Slurm advertises the entire accelerator node as one CPU slot. Detach PRTE from
# the batch allocation so tt-run's explicit hostfile and rank bindings control
# placement of both device ranks within the node.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    # shellcheck disable=SC2046
    unset $(env | sed -n 's/^\(SLURM[^=]*\)=.*/\1/p')
    export PRTE_MCA_ras="^slurm"
    export PRTE_MCA_plm="^slurm"
fi

# HostWeightBridge is pure MPI, so use two independent 1x1 meshes and run
# without the rollout test conftest that enables FABRIC_2D.
cd "${TEST_DIR}"
export PYTHONPATH="${TT_METAL_HOME}/tt-train/sources/examples/grpo_remote_rollout:${TEST_DIR}:${TT_METAL_HOME}:${PYTHONPATH:-}"

CMD="python3 -m pytest -s -p no:cacheprovider --noconftest --rootdir=${TEST_DIR} ${TEST_FILE}"

"${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py" \
    --rank-binding "${RANK_BINDINGS_FILE}" \
    --mpi-args "--hostfile ${HOST_FILE} --tag-output --oversubscribe" \
    -- ${CMD}
