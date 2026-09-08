#!/bin/bash
set -euo pipefail
: "${TT_METAL_HOME:?TT_METAL_HOME is not set}"
EX_DIR="${TT_METAL_HOME}/tt-train/sources/examples/grpo_remote_rollout/gsm8k_fully_async"
cd "${EX_DIR}"
unset $(env | sed -n 's/^\(SLURM[^=]*\)=.*/\1/p')
export PRTE_MCA_ras="^slurm"
export PRTE_MCA_plm="^slurm"
exec "${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py" \
    --rank-binding "${EX_DIR}/configurations/independent_1x1/rank_bindings.yaml" \
    --mpi-args "--hostfile ${EX_DIR}/configurations/independent_1x1/hosts.txt --tag-output --oversubscribe" \
    python3 "${EX_DIR}/gsm8k_fully_async_training_example.py"
