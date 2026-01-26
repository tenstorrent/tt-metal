#!/bin/bash
#SBATCH --job-name=sys_health
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --partition=wh_cluster
#SBATCH --nodelist=wh-glx-a03u02,wh-glx-a04u08
#SBATCH --output=sys_health_%j.out
#SBATCH --error=sys_health_%j.err
#SBATCH --exclusive

export TT_METAL_HOME="/data/dmadic/tt-metal"
cd ${TT_METAL_HOME}

echo "Running system health test on each node..."
echo ""

srun --ntasks-per-node=1 bash -c '
    hostname=$(hostname)
    echo "Running test on ${hostname}..."
    ./build/test/tt_metal/tt_fabric/test_system_health > /data/dmadic/system_health_${hostname}.txt 2>&1
    echo "Saved output to /data/dmadic/system_health_${hostname}.txt"
'

echo ""
echo "Done. Check /data/dmadic/system_health_*.txt files"
