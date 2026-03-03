#!/bin/bash
#SBATCH --partition=bh_pod_4x32_C12
#SBATCH --nodes=4
#SBATCH --nodelist=bh-glx-c01u02,bh-glx-c02u02,bh-glx-c02u08,bh-glx-c01u08
#SBATCH --job-name=pipeline_parallel_training_batch_1
#SBATCH --output=pipeline_parallel_training_%j.out
#SBATCH --error=pipeline_parallel_training_%j.err

# Note: Manually set the workload here
WORKLOAD="llama405b"

# Common environmental variables
if [ -z "${TT_METAL_HOME:-}" ]; then
    TT_METAL_HOME="/data/${USER}/tt-metal"
fi
export TT_METAL_HOME
export PYTHONPATH="${TT_METAL_HOME}"
source ${TT_METAL_HOME}/python_env/bin/activate
export LD_LIBRARY_PATH="/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH"
export TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS=120000
PP_ROOT="${TT_METAL_HOME}/tt-train/sources/examples/python/multihost/pipeline_parallel_training"

# Create hostfile based on SLURM nodes
HOSTFILE="/tmp/hostfile_${SLURM_JOB_ID}"
scontrol show hostnames $SLURM_JOB_NODELIST | while read host; do
    echo "${host} slots=1"
done > ${HOSTFILE}

# Select config and host config based on workload
if [[ "$WORKLOAD" == "llama8b" ]]; then
    CONFIG_FILE="training_configs/training_shakespeare_llama8b_intrahost_pp4.yaml"
    HOST_CONFIG="pp4_galaxy"
elif [[ "$WORKLOAD" == "llama70b_4stage" ]]; then
    CONFIG_FILE="training_configs/training_shakespeare_llama70b_pp4_tp32_fabric_galaxy.yaml"
    HOST_CONFIG="pp4_galaxy"
elif [[ "$WORKLOAD" == "llama70b_16stage" ]]; then
    CONFIG_FILE="training_configs/training_shakespeare_llama70b_pp16_fabric_galaxy.yaml"
    HOST_CONFIG="pp16_galaxy"
elif [[ "$WORKLOAD" == "llama405b" ]]; then
    CONFIG_FILE="training_configs/training_shakespeare_llama405b_pp_fabric.yaml"
    HOST_CONFIG="pp16_galaxy"
else
    echo "Unknown workload: $WORKLOAD"
    exit 1
fi

# copy rankfile to /tmp so there are no dashes in the name (workaround for parsing error)
cp ${PP_ROOT}/configurations/${HOST_CONFIG}/rankfile.txt /tmp/rankfile.txt

# run training
tt-run --verbose \
    --mpi-args "--hostfile ${HOSTFILE} --mca btl_tcp_if_exclude docker0,lo  --mca btl self,tcp --tag-output --oversubscribe --map-by rankfile:file=/tmp/rankfile.txt" \
    --rank-binding ${PP_ROOT}/configurations/${HOST_CONFIG}/rank_bindings.yaml \
    python ${PP_ROOT}/training.py -c ${CONFIG_FILE}
