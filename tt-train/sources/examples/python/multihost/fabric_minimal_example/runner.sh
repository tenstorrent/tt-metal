#!/bin/bash
#SBATCH --partition=bh_pod_4x32_1
#SBATCH --nodelist=bh-glx-b04u02,bh-glx-b04u08
#SBATCH --job-name=fabric_minimal
#SBATCH --output=fabric_minimal_%j.out
#SBATCH --error=fabric_minimal_%j.err

# Set environmental variables
export TT_METAL_HOME="/data/${USER}/pr_review/tt-metal"
export PYTHONPATH="${TT_METAL_HOME}:${PYTHONPATH}"
source ${TT_METAL_HOME}/python_env/bin/activate
export LD_LIBRARY_PATH="/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH"

FABRIC_MINIMAL="/data/${USER}/pr_review/tt-metal/tt-train/sources/examples/python/multihost/fabric_minimal_example"
CONFIG_FILE="training_configs/training_shakespeare_tinyllama_2tier_fabric.yaml"

HOSTLIST="/data/jmalone/config/2loudboxes/hosts.txt"
RANK_BINDINGS="/data/jmalone/config/2loudboxes/rank_bindings.yaml"

tt-run --mpi-args "--map-by rankfile:file=${HOSTLIST} -x TT_METAL_HOME --mca mpi_show_mca_params all --mca btl_tcp_if_include eno1 --mca oob_tcp_if_include eno1 --mca btl self,tcp --tag-output" --rank-binding ${RANK_BINDINGS} python ${FABRIC_MINIMAL}/example.py -c ${CONFIG_FILE}
