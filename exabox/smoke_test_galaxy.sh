#!/bin/bash
#SBATCH --job-name=galaxy_smoke
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --partition=wh_pod_8x16_1
#SBATCH --nodelist=wh-glx-a03u08,wh-glx-a04u08
#SBATCH --output=galaxy_smoke_%j.out
#SBATCH --error=galaxy_smoke_%j.err
#SBATCH --exclusive

# Galaxy-to-Galaxy Smoke Test
# Tests basic connectivity between two Tenstorrent Galaxy systems

echo "=========================================="
echo "Galaxy-to-Galaxy Smoke Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo ""

# Set up environment
export TT_METAL_HOME="/data/dmadic/tt-metal"
export PYTHONPATH="${TT_METAL_HOME}"

cd ${TT_METAL_HOME}

# Activate Python environment
source python_env/bin/activate

echo "TT_METAL_HOME: ${TT_METAL_HOME}"
echo "Python: $(which python)"
echo ""

# Create hostfile from Slurm allocation
HOSTFILE="/tmp/hostfile_${SLURM_JOB_ID}"
scontrol show hostnames $SLURM_JOB_NODELIST | while read host; do
    echo "${host} slots=1"
done > ${HOSTFILE}

echo "MPI Hostfile (${HOSTFILE}):"
cat ${HOSTFILE}
echo ""

# Run the smoke test
# Key: --mca btl_tcp_if_exclude docker0,lo prevents MPI from using
# local-only interfaces for cross-host communication
echo "Running Galaxy-to-Galaxy smoke test..."
echo ""

tt-run --verbose \
    --rank-binding tests/tt_metal/distributed/config/exabox_2_galaxy_rank_binding.yaml \
    --mpi-args "--hostfile ${HOSTFILE} --mca btl_tcp_if_exclude docker0,lo --tag-output" \
    python tests/ttnn/distributed/smoke_test_galaxy_to_galaxy.py

EXIT_CODE=$?

rm -f ${HOSTFILE}

echo ""
echo "=========================================="
echo "Test completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
