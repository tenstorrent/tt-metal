#!/bin/bash
#SBATCH --partition=bh_pod_4x32_B89
#SBATCH --nodes=2
#SBATCH --overcommit
#SBATCH --nodelist=bh-glx-b08u08,bh-glx-b09u08
#SBATCH --job-name=report_links
#SBATCH --output=report_links_%j.out
#SBATCH --error=report_links_%j.err

# Set environmental variables
export TT_METAL_HOME="/data/${USER}/tt-metal"
export TT_METAL_RUNTIME_ROOT="/data/${USER}/tt-metal"
export PYTHONPATH="${TT_METAL_HOME}"
source ${TT_METAL_HOME}/python_env/bin/activate
export LD_LIBRARY_PATH="/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH"

# Reset devices
# srun tt-smi -glx_reset

# Create hostfile
HOSTFILE="/tmp/hostfile_${SLURM_JOB_ID}"
scontrol show hostnames $SLURM_JOB_NODELIST | while read host; do
    echo "${host} slots=1"
done > ${HOSTFILE}

cat ${HOSTFILE}

# Set rank binding configuration
RANK_BINDING="${TT_METAL_HOME}/rank_bindings.yaml"

# Run the test
srun --ntasks-per-node=1 python3 tests/tt_metal/tt_fabric/utils/generate_rank_bindings_to_dir.py

# combine them (only runs on host 0)
python tests/tt_metal/tt_fabric/utils/combine_rank_bindings.py \
    bh-glx-b08u08,bh-glx-b09u08 \
    --output-dir combined/ \
    --cluster-config /data/scaleout_configs/bh_glx_exabox/cabling_descriptor.textproto \
    --remap-to-ring
