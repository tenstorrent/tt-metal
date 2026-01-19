#!/bin/bash

# Script to recover 8x16 cluster (distributed tt-smi reset + cluster validation)
#
# Usage: ./recover_8x16.sh <comma-separated-host-list>
# Example: ./recover_8x16.sh bh-glx-c01u02,bh-glx-c01u08,bh-glx-c02u02,bh-glx-c02u08

# Check if host list is provided
if [ $# -eq 0 ]; then
    echo "Error: No host list provided"
    echo "Usage: $0 <comma-separated-host-list>"
    echo "Example: $0 bh-glx-c01u02,bh-glx-c01u08,bh-glx-c02u02,bh-glx-c02u08"
    exit 1
fi

# Get host list from command line argument
HOSTS="$1"

echo "Using hosts: $HOSTS"
echo ""

echo "Running tt-smi -glx_reset..."
mpirun --host $HOSTS --mca btl_tcp_if_exclude docker0,lo tt-smi -glx_reset
sleep 5

echo ""
echo "Running cluster validation..."
mpirun --host $HOSTS --mca btl_tcp_if_exclude docker0,lo --tag-output ./build/tools/scaleout/run_cluster_validation --factory-descriptor-path /data/scaleout_configs/5xBH_8x16_intrapod/fsd.textproto --send-traffic --num-iterations 5
