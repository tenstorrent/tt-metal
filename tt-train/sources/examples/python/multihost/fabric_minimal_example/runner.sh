#!/bin/bash
#SBATCH --partition=debug
#SBATCH --job-name=fabric_minimal
#SBATCH --output=fabric_minimal_%j.out
#SBATCH --error=fabric_minimal_%j.err

# Set environmental variables
export TT_METAL_HOME="/data/${USER}/tt-metal"
export PYTHONPATH="${TT_METAL_HOME}:${PYTHONPATH}"
source ${TT_METAL_HOME}/python_env/bin/activate
export LD_LIBRARY_PATH="/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH"

FABRIC_MINIMAL="/data/${USER}/tt-metal/tt-train/sources/examples/python/multihost/fabric_minimal_example"
CONFIG_FILE="training_configs/training_shakespeare_tinyllama_2tier_fabric.yaml"

# Create hostfile from Slurm allocation
HOSTFILE="/tmp/hostfile_${SLURM_JOB_ID}"
scontrol show hostnames $SLURM_JOB_NODELIST | while read host; do
    echo "${host} slots=1"
done > ${HOSTFILE}

tt-run --mpi-args "--hostfile ${HOSTFILE} -x TT_METAL_HOME --mca mpi_show_mca_params all --mca btl_tcp_if_include eno1 --mca oob_tcp_if_include eno1 --mca btl self,tcp --tag-output" --rank-binding ${FABRIC_MINIMAL}/configurations/2loudboxes/rank_bindings.yaml python ${FABRIC_MINIMAL}/example.py -c ${CONFIG_FILE}
