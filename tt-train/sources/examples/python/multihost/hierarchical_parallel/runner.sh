#!/bin/bash

#SBATCH --partition=debug
#SBATCH --job-name=hierarchical_parallel_training
#SBATCH --output=hierarchical_parallel_training_%j.out
#SBATCH --error=hierarchical_parallel_training_%j.err
# Set environmental variables

export TT_METAL_HOME="/data/${USER}/tt-metal"
export PYTHONPATH="${TT_METAL_HOME}:${PYTHONPATH}"
source ${TT_METAL_HOME}/python_env/bin/activate
export TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS=120000
export HIERARCHICAL_ROOT="/data/${USER}/tt-metal/tt-train/sources/examples/python/multihost/hierarchical_parallel"
export LD_LIBRARY_PATH="/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH"

CONFIG_FILE="training_configs/training_shakespeare_tinyllama_3tier_fabric.yaml"
HOST_CONFIG="5loudboxes"

tt-run --mpi-args "--hostfile ${HIERARCHICAL_ROOT}/configurations/${HOST_CONFIG}/hosts.txt -x TT_METAL_HOME --mca mpi_show_mca_params all --mca btl_tcp_if_include eno1 --mca oob_tcp_if_include eno1 --mca btl self,tcp --tag-output" --rank-binding ${HIERARCHICAL_ROOT}/configurations/${HOST_CONFIG}/rank_bindings.yaml python ${TT_METAL_HOME}/tt-train/sources/examples/python/multihost/hierarchical_parallel/training.py -c ${CONFIG_FILE}
