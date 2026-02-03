#!/bin/bash

#SBATCH --partition=bh_pod_4x32_2
#SBATCH --nodes=4
#SBATCH --nodelist=bh-glx-b08u02,bh-glx-b08u08,bh-glx-b09u08,bh-glx-b09u02
#SBATCH --job-name=hierarchical_parallel_training
#SBATCH --output=hierarchical_parallel_training_%j.out
#SBATCH --error=hierarchical_parallel_training_%j.err
# Set environmental variables

export TT_METAL_HOME="/data/${USER}/pr_review/tt-metal"
export PYTHONPATH="${TT_METAL_HOME}"
source ${TT_METAL_HOME}/python_env/bin/activate
export TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS=120000
export HIERARCHICAL_ROOT="/data/${USER}/pr_review/tt-metal/tt-train/sources/examples/python/multihost/hierarchical_parallel"
export LD_LIBRARY_PATH="/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH"

HOSTFILE="/tmp/hostfile_${SLURM_JOB_ID}"
scontrol show hostnames $SLURM_JOB_NODELIST | while read host; do
    echo "${host} slots=1"
done > ${HOSTFILE}

CONFIG_FILE="training_configs/training_shakespeare_tinyllama_3tier_fabric.yaml"
HOST_CONFIG="4galaxies"

RANKFILE="/data/jmalone/rankfile.txt"

tt-run --mpi-args "--hostfile ${HOSTFILE} --mca btl_tcp_if_exclude docker0,lo  --mca btl self,tcp --tag-output --map-by rankfile:file=${RANKFILE}" \
    --rank-binding ${HIERARCHICAL_ROOT}/configurations/${HOST_CONFIG}/rank_bindings.yaml \
    python ${TT_METAL_HOME}/tt-train/sources/examples/python/multihost/hierarchical_parallel/training.py -c ${CONFIG_FILE}
