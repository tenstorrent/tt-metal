#!/bin/bash
#SBATCH --partition=bh_sp_5x4x32_C1_C10
#SBATCH --nodes=4
#SBATCH --nodelist=bh-glx-c01u02,bh-glx-c01u08,bh-glx-c02u08,bh-glx-c02u02
#SBATCH --job-name=generate_cluster_mgd_and_bindings
#SBATCH --output=generate_cluster_mgd_and_bindings_%j.out
#SBATCH --error=generate_cluster_mgd_and_bindings_%j.err

# Set environmental variables
export TT_METAL_HOME="/data/${USER}/tt-metal"
export TT_METAL_RUNTIME_ROOT="${TT_METAL_HOME}"
export PYTHONPATH="${TT_METAL_HOME}"
source ${TT_METAL_HOME}/python_env/bin/activate
export LD_LIBRARY_PATH="/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH"

# Reset devices
srun tt-smi -glx_reset

# Create hostfile
HOSTFILE="/tmp/hostfile_${SLURM_JOB_ID}"
scontrol show hostnames $SLURM_JOB_NODELIST | while read host; do
    echo "${host} slots=1"
done > ${HOSTFILE}

cat ${HOSTFILE}

# Generate rank bindings for each host
srun --ntasks-per-node=1 python3 generate_rank_bindings_to_dir.py

# combine them (only runs on host 0), order of hosts must match order of desired connections
python3 combine_rank_bindings.py \
    "bh-glx-c01u02,bh-glx-c01u08,bh-glx-c02u08,bh-glx-c02u02" \
    --output-dir combined/ \
    --cluster-config /data/scaleout_configs/bh_glx_exabox/cabling_descriptor.textproto \
    --remap-to-ring
