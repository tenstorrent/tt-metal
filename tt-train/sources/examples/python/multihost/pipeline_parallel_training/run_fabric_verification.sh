#!/bin/bash
#SBATCH --job-name=ttnn_fabric_verification
#SBATCH --output=ttnn_fabric_verification_%j.out
#SBATCH --error=ttnn_fabric_verification_%j.err

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

# 1 rank per host (each rank opens a 1x8 mesh = one galaxy)
RANKS_PER_HOST="1"

# generate rankfile to /tmp with unique name (workaround for parsing error with dashes in paths)
RANKFILE="/tmp/rankfile_${USER}_${SLURM_JOB_ID}.txt"
scontrol show hostnames | python ${PP_ROOT}/make_rankfile.py -n ${RANKS_PER_HOST} -o ${RANKFILE}

# run fabric verification
tt-run --verbose \
    --mpi-args "--oversubscribe --map-by rankfile:file=${RANKFILE}" \
    --rank-binding ${PP_ROOT}/configurations/2galaxy_verification/rank_bindings.yaml \
    python ${PP_ROOT}/ttnn_fabric_verification.py
