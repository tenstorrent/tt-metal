#!/bin/bash
#SBATCH --nodes=5
#SBATCH --nodelist=metal-wh-18,metal-wh-09,metal-wh-10,metal-wh-11,metal-wh-12
#SBATCH --partition=debug
#SBATCH --job-name=pipeline_parallel_training
#SBATCH --output=pipeline_parallel_training_%j.out
#SBATCH --error=pipeline_parallel_training_%j.err
# Set environmental variables
export TT_METAL_HOME="/data/${USER}/tt-metal"
export PYTHONPATH="${TT_METAL_HOME}:${PYTHONPATH}"
source ${TT_METAL_HOME}/python_env/bin/activate

CONFIG_FILE="training_configs/training_shakespeare_llama7b_pp_fabric.yaml"

tt-run --mpi-args "-x TT_METAL_HOME --mca mpi_show_mca_params all --mca btl_tcp_if_include eno1 --mca oob_tcp_if_include eno1 --mca btl self,tcp --tag-output" --rank-binding configurations/5loudboxes/rank_bindings.yaml python ${TT_METAL_HOME}/tt-train/sources/examples/python/multihost/pipeline_parallel_training/training.py -c ${CONFIG_FILE}
