#!/bin/bash
#SBATCH --nodes=5
#SBATCH --partition=debug
#SBATCH --job-name=test
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err

# Set environmental variables
export TT_MESH_ID=0
export TT_MESH_HOST_RANK=0

export TT_METAL_HOME="/data/rreece/tt-metal"
#export PYTHONPATH="${TT_METAL_HOME}:${PYTHONPATH}"
#source ${TT_METAL_HOME}/python_env/bin/activate

#srun --mpi=pmix hostname
#srun hostname
mpirun hostname
