#!/bin/bash
#SBATCH --job-name=pipeline_parallel_training_batch_1
#SBATCH --output=pipeline_parallel_training_%j.out
#SBATCH --error=pipeline_parallel_training_%j.err

# Note: Manually set the workload here
WORKLOAD="llama70b_16stage"

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

# Make sure all galaxies are in a good state
srun tt-smi -glx_reset

# Select config and host config based on workload
RANKS_PER_HOST="4"
if [[ "$WORKLOAD" == "llama8b" ]]; then
    CONFIG_FILE="training_configs/training_shakespeare_llama8b_intrahost_pp4.yaml"
    HOST_CONFIG="1galaxy_pp4"
elif [[ "$WORKLOAD" == "llama70b_4stage" ]]; then
    CONFIG_FILE="training_configs/training_shakespeare_llama70b_pp4_tp32_fabric_galaxy.yaml"
    HOST_CONFIG="4galaxy_pp4"
    RANKS_PER_HOST="1"
elif [[ "$WORKLOAD" == "llama70b_16stage" ]]; then
    CONFIG_FILE="training_configs/training_shakespeare_llama70b_pp16_fabric_galaxy.yaml"
    HOST_CONFIG="4galaxy_pp16"
elif [[ "$WORKLOAD" == "llama405b" ]]; then
    CONFIG_FILE="training_configs/training_shakespeare_llama405b_pp_fabric.yaml"
    HOST_CONFIG="4galaxy_pp16"
else
    echo "Unknown workload: $WORKLOAD"
    exit 1
fi

# generate rankfile to /tmp so there are no dashes in the name (workaround for parsing error)
scontrol show hostnames | python make_rankfile.py -n ${RANKS_PER_HOST} -o /tmp/rankfile.txt

# run training
tt-run --verbose \
    --mpi-args "--oversubscribe --map-by rankfile:file=/tmp/rankfile.txt" \
    --rank-binding ${PP_ROOT}/configurations/${HOST_CONFIG}/rank_bindings.yaml \
    python ${PP_ROOT}/training.py -c ${CONFIG_FILE}
