#!/bin/bash
#SBATCH --job-name=reset_devs
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --partition=wh_pod_8x16_1
#SBATCH --nodelist=wh-glx-a03u08,wh-glx-a04u08
#SBATCH --output=reset_devs_%j.out
#SBATCH --error=reset_devs_%j.err
#SBATCH --exclusive

# Reset Tenstorrent devices on the two Galaxy hosts only

echo "=========================================="
echo "Resetting TT Devices on 2 Galaxies"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo ""

# Run tt-smi -r on each node using srun
echo "Resetting devices on all allocated nodes..."
srun --ntasks-per-node=1 tt-smi -r

echo ""
echo "Waiting for devices to stabilize..."
sleep 5

echo ""
echo "Verifying device status on all nodes..."
srun --ntasks-per-node=1 tt-smi -s

echo ""
echo "=========================================="
echo "Reset completed"
echo "End time: $(date)"
echo "=========================================="
