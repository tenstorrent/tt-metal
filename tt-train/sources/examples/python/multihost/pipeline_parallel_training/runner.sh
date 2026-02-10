#!/bin/bash
#SBATCH --partition=debug
#SBATCH --job-name=pipeline_parallel_training
#SBATCH --output=pipeline_parallel_training_%j.out
#SBATCH --error=pipeline_parallel_training_%j.err
# Set environmental variables
export TT_METAL_HOME="/data/${USER}/tt-metal"
export PYTHONPATH="${TT_METAL_HOME}:${PYTHONPATH}"
source ${TT_METAL_HOME}/python_env/bin/activate
export LD_LIBRARY_PATH="/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH"

CONFIG_FILE="training_configs/training_shakespeare_llama7b_pp_fabric.yaml"
PP_ROOT="${TT_METAL_HOME}/tt-train/sources/examples/python/multihost/pipeline_parallel_training"
HOST_CONFIG="5loudboxes"

tt-run --mpi-args "--hostfile ${PP_ROOT}/configurations/${HOST_CONFIG}/hosts.txt -x TT_METAL_HOME --mca mpi_show_mca_params all --mca btl_tcp_if_include eno1 --mca oob_tcp_if_include eno1 --mca btl self,tcp --tag-output" --rank-binding ${PP_ROOT}/configurations/${HOST_CONFIG}/rank_bindings.yaml python ${PP_ROOT}/training.py -c ${CONFIG_FILE}
