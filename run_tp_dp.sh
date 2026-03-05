# #!/bin/bash
# #SBATCH --partition=bh_sp_5x4x32_C1_C10
# #SBATCH --nodelist=bh-glx-c01u08,bh-glx-c05u08
# #SBATCH --nodes=2
# #SBATCH --job-name=run_tp_dp
# #SBATCH --output=run_tp_dp_%j.out
# #SBATCH --error=run_tp_dp_%j.err

# # Set environmental variables
# export TT_METAL_HOME="/data/${USER}/llama/tt-metal"
# export TT_METAL_RUNTIME_ROOT="${TT_METAL_HOME}"
# export PYTHONPATH="${TT_METAL_HOME}"
# source ${TT_METAL_HOME}/python_env/bin/activate
# export LD_LIBRARY_PATH="/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH"
# export TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS=120000
# export TT_MESH_GRAPH_DESC_PATH="${TT_METAL_HOME}/dual_glx_4x8_mgd.textproto"

# #srun tt-smi -glx_reset

# HOSTFILE="/tmp/hostfile_${SLURM_JOB_ID}"
# scontrol show hostnames $SLURM_JOB_NODELIST | while read host; do
#     echo "${host} slots=1"
# done > ${HOSTFILE}

# tt-run --verbose \
#     --mpi-args "--hostfile ${HOSTFILE} --mca btl_tcp_if_exclude docker0,lo  --mca btl self,tcp --tag-output" \
#     --rank-binding /data/jmalone/llama/tt-metal/dual_rank_binding.yaml \
#     ../build/tt-train/sources/examples/nano_gpt/nano_gpt \
#       --config configs/training_configs/training_llama8b_tp_ddp_pp_galaxy.yaml


# Run baremetal

HOST_LIST="bh-glx-b08u02,bh-glx-b08u08"

# Set environmental variables
export TT_METAL_HOME="/data/${USER}/llama/tt-metal"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_HOME}"
export PYTHONPATH="${TT_METAL_HOME}"
source ${TT_METAL_HOME}/python_env/bin/activate
export LD_LIBRARY_PATH="/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH"
export TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS=120000

SLURM_JOB_ID="BAREMETAL"
CONFIG="SINGLE"


if [ "$CONFIG" == "DUAL" ]; then
    HOSTFILE="/tmp/hostfile_${SLURM_JOB_ID}"

    echo "bh-glx-b08u02 slots=1" > ${HOSTFILE}
    echo "bh-glx-b08u08 slots=1" >> ${HOSTFILE}

    tt-run --verbose \
        --mpi-args "--hostfile ${HOSTFILE} --mca btl_tcp_if_exclude docker0,lo  --mca btl self,tcp --tag-output" \
        --rank-binding /data/jmalone/llama/tt-metal/dual_rank_binding.yaml \
        ../build/tt-train/sources/examples/nano_gpt/nano_gpt \
        --config configs/training_configs/training_llama8b_tp_ddp_pp_galaxy.yaml
    exit 0
fi

export TT_MESH_GRAPH_DESC_PATH="${TT_METAL_HOME}/galaxy_4x8_mgd.textproto"
../build/tt-train/sources/examples/nano_gpt/nano_gpt \
    --config configs/training_configs/training_llama8b_tp_ddp_galaxy.yaml
