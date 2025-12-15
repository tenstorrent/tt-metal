#!/bin/bash

#SBATCH --partition=debug
#SBATCH --job-name=pipeline_parallel_training
#SBATCH --output=pipeline_parallel_training_%j.out
#SBATCH --error=pipeline_parallel_training_%j.err
# Set environmental variables

export TT_METAL_HOME="/data/${USER}/tt-metal"
export PYTHONPATH="${TT_METAL_HOME}:${PYTHONPATH}"
source ${TT_METAL_HOME}/python_env/bin/activate

CONFIG_FILE="training_configs/training_shakespeare_nanogpt_2tier_fabric.yaml"

tt-run --mpi-args "--rankfile /data/slurm/rankfile_${SLURM_JOB_ID}.txt -x TT_METAL_HOME --mca mpi_show_mca_params all --mca btl_tcp_if_include eno1 --mca oob_tcp_if_include eno1 --mca btl self,tcp --tag-output" --rank-binding /data/slurm/rank_bindings_${SLURM_JOB_ID}.yaml python ${TT_METAL_HOME}/tt-train/sources/examples/python/multihost/hierarchical_parallel/training.py -c ${CONFIG_FILE}
