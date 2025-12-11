#!/bin/bash
#SBATCH --nodes=2
#SBATCH --partition=debug
#SBATCH --job-name=test
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err

# Set environmental variables
export TT_METAL_HOME="/data/rreece/tt-metal"
export PYTHONPATH="${TT_METAL_HOME}:${PYTHONPATH}"
source ${TT_METAL_HOME}/python_env/bin/activate

tt-run --mpi-args "--verbose -x TT_METAL_HOME --mca mpi_show_mca_params all --mca btl_tcp_if_include eno1 --mca oob_tcp_if_include eno1 --mca btl self,tcp --tag-output" --rank-binding rank_bindings.yaml python ${TT_METAL_HOME}/tt-train/sources/examples/python/multihost/mpi_minimal_example.py
