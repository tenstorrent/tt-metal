#!/bin/bash
#SBATCH --nodes=2
#SBATCH --nodelist=metal-wh-09,metal-wh-18
#SBATCH --partition=debug
#SBATCH --job-name=test
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err

# Set environmental variables
export TT_METAL_HOME="/data/abustamante/tt-metal"
export PYTHONPATH="${TT_METAL_HOME}:${PYTHONPATH}"
source ${TT_METAL_HOME}/python_env/bin/activate

CONFIG_FILE="training_configs/training_shakespeare_tinyllama_tensor_parallel_3tier_fabric.yaml"

tt-run --mpi-args "-x TT_METAL_HOME --mca mpi_show_mca_params all --mca btl_tcp_if_include eno1 --mca oob_tcp_if_include eno1 --mca btl self,tcp --tag-output" --rank-binding configurations/2loudboxes/rank_bindings.yaml python ${TT_METAL_HOME}/tt-train/sources/examples/python/multihost/fabric_minimal_example/example.py -c ${CONFIG_FILE}
