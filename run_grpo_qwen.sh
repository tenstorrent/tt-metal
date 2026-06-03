#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=bh_sp5_aisle_c_partial
#SBATCH --nodelist=bh-glx-c04u08
#SBATCH --time=UNLIMITED
#SBATCH --job-name=run_grpo_qwen
#SBATCH --output=/data/awliu/run_grpo/grpo_qwen_glx_%j.out
#SBATCH --error=/data/awliu/run_grpo/grpo_qwen_glx_%j.err

# GRPO fine-tuning of Qwen3-32B on BoolQ (tensor-parallel).
#
# Qwen3-32B is sharded across the device mesh via tensor parallelism (TP=8,
# since it has 8 KV heads). The default config (grpo_boolq_qwen_32b.yaml) uses
# mesh_shape [4, 8] = 32 devices, which matches the single BlackHole galaxy
# descriptor set below. The mesh_shape in the YAML MUST match this descriptor.

# Set environmental variables
export HOME="/data/${USER}"
export TT_METAL_HOME="/data/${USER}/tt-metal-new"
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME
export TT_METAL_CACHE=/data/awliu/.cache/tt-metal-cache-job2

export PYTHONPATH="${TT_METAL_HOME}"
source ${TT_METAL_HOME}/python_env/bin/activate
pip install wandb
export LD_LIBRARY_PATH="/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH"
# Single BH Galaxy (4x8) mesh graph descriptor — matches mesh_shape [4, 8].
export TT_MESH_GRAPH_DESC_PATH="${TT_METAL_HOME}/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto"

export HF_HUB_DISABLE_PROGRESS_BARS=1
# HF_TOKEN lives in a chmod 600 file so it never lands in the script,
# job env dumps, or the tt-metal git worktree. (Qwen3 is NOT a gated repo, so
# a token is not strictly required, but kept for parity / private mirrors.)
if [ -f /data/awliu/.hf_env ]; then
    source /data/awliu/.hf_env
else
    echo "WARNING: /data/awliu/.hf_env not found; HF auth (if needed) will fail." >&2
fi


# Weights & Biases (online). Set WANDB_MODE=offline on hosts without internet
# and `wandb sync` afterwards from a host with outbound access.
export WANDB_MODE=online
export WANDB_DIR=/data/awliu/run_grpo/wandb
mkdir -p "$WANDB_DIR"
# WANDB_API_KEY lives in a chmod 600 file so it never lands in the script,
# job env dumps, or the tt-metal git worktree.
if [ -f /data/awliu/.wandb_env ]; then
    source /data/awliu/.wandb_env
else
    echo "WARNING: /data/awliu/.wandb_env not found; W&B will fail to authenticate." >&2
fi

# Reset devices (galaxy-wide reset on BH Galaxy nodes)
srun /usr/local/bin/tt-smi -glx_reset

ulimit -c 0
ulimit -t unlimited 2>/dev/null || ulimit -t hard

echo "=== /proc/self/limits ===" >&2
cat /proc/self/limits >&2
echo "=== hostname=$(hostname) job=$SLURM_JOB_ID ===" >&2

python -u tt-train/sources/examples/grpo/boolq_training_example.py \
    --model qwen \
    --model_id "Qwen/Qwen3-32B" \
    --config "${TT_METAL_HOME}/tt-train/configs/training_configs/grpo_boolq_qwen_32b.yaml" \
    --wandb \
    --wandb_run_name "grpo_qwen32b_1"
