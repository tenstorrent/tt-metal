#!/bin/bash
#SBATCH --partition=bh_pod_4x32_2
#SBATCH --nodes=4
#SBATCH --nodelist=bh-glx-b08u02,bh-glx-b08u08,bh-glx-b09u08,bh-glx-b09u02
#SBATCH --job-name=pipeline_parallel_training
#SBATCH --output=pipeline_parallel_training_%j.out
#SBATCH --error=pipeline_parallel_training_%j.err

# Set environmental variables
export TT_METAL_HOME="/data/${USER}/pr_review/tt-metal"
export PYTHONPATH="${TT_METAL_HOME}"
source ${TT_METAL_HOME}/python_env/bin/activate
export LD_LIBRARY_PATH="/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH"

#srun tt-smi -r
#srun tt-smi -glx_reset

HOSTFILE="/tmp/hostfile_${SLURM_JOB_ID}"
scontrol show hostnames $SLURM_JOB_NODELIST | while read host; do
    echo "${host} slots=1"
done > ${HOSTFILE}

CONFIG_FILE="training_configs/training_shakespeare_llama7b_pp_fabric.yaml"
PP_ROOT="${TT_METAL_HOME}/tt-train/sources/examples/python/multihost/pipeline_parallel_training"
HOST_CONFIG="4galaxies"

#HOSTFILE="${PP_ROOT}/configurations/${HOST_CONFIG}/hosts.txt"
RANKFILE="/data/jmalone/rankfile.txt"

tt-run --verbose \
    --mpi-args "--hostfile ${HOSTFILE} --mca btl_tcp_if_exclude docker0,lo  --mca btl self,tcp --tag-output --map-by rankfile:file=${RANKFILE}" \
    --rank-binding ${PP_ROOT}/configurations/${HOST_CONFIG}/rank_bindings.yaml \
    python ${PP_ROOT}/training.py -c ${CONFIG_FILE}
