#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Launches the 2-rank GRPO BoolQ training example via tt-run.
#
# Topology (mirrors tests/weight_transfer/runner.sh):
#   Rank 0 (TTML): owns one N300 board, opens a [1, 2] mesh, drives
#                  training via ttml + GRPOTrainer.
#   Rank 1 (TTT):  owns one N300 board, opens a [1, 1] mesh, hosts
#                  the tt-transformers generation worker over RPC.
#
# tt-run wraps mpirun:
#   --rank-binding   maps each MPI rank to a (mesh_id, mesh_host_rank)
#                    and sets per-rank env (e.g. TT_VISIBLE_DEVICES).
#   --mpi-args       passed straight to mpirun. We use --hostfile here
#                    so both ranks land on localhost.
#
# Override config locations with --rank-bindings / --hostfile if you
# adapt this to a multi-host or larger-mesh setup.

set -euo pipefail

if [[ -z "${TT_METAL_HOME:-}" ]]; then
    echo "TT_METAL_HOME is not set" >&2
    exit 1
fi

EX_DIR="${TT_METAL_HOME}/tt-train/sources/examples/grpo/boolq"
HOST_FILE="${EX_DIR}/configurations/local2/hosts.txt"
RANK_BINDINGS_FILE="${EX_DIR}/configurations/local2/rank_bindings.yaml"
SCRIPT="${EX_DIR}/boolq_training_example.py"

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --hostfile)
            shift; HOST_FILE="$1" ;;
        --rank-bindings
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

# tt-run resolves the relative `mesh_graph_desc_path` inside
# rank_bindings.yaml against (1) TT_METAL_HOME, (2) the launch
# directory (cwd at tt-run invocation), and (3) cwd at resolution
# time -- not against the rank_bindings file's own directory. cd into
# the example dir so "configurations/local2/mgd.textproto" resolves
# here.
cd "${EX_DIR}"

CMD="python3 ${SCRIPT}"

"${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py" \
    --rank-binding "${RANK_BINDINGS_FILE}" \
    --mpi-args "--hostfile ${HOST_FILE} --tag-output" \
    ${CMD}
