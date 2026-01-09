#!/bin/bash

# Script to run cluster validation commands for 50 iterations
# Each iteration's output is logged to a separate file
#
# Usage: ./run_validation_8x16.sh <comma-separated-host-list>
# Example: ./run_validation_8x16.sh bh-glx-c01u02,bh-glx-c01u08,bh-glx-c02u02,bh-glx-c02u08

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

# Create output directory if it doesn't exist
mkdir -p validation_output

for i in {1..1}; do
    echo "Starting iteration $i of 50..."

    LOG_FILE="validation_output/cluster_validation_iteration_${i}.log"

    {
        echo "=========================================="
        echo "Iteration: $i"
        echo "Timestamp: $(date)"
        echo "=========================================="
        echo ""

        echo "Running tt-smi -glx_reset..."
        mpirun --host $HOSTS --mca btl_tcp_if_include ens5f0np0 tt-smi -glx_reset
        sleep 5

        echo ""
        echo "Running cluster validation..."
        mpirun --host $HOSTS --mca btl_tcp_if_include ens5f0np0 --tag-output ./build_Release/tools/scaleout/run_cluster_validation --factory-descriptor-path /data/local-syseng-manual/5x8x16_fsd.textproto --send-traffic --num-iterations 5
        echo "Iteration $i completed at $(date)"
        echo "=========================================="
    } 2>&1 | tee "$LOG_FILE"

    echo "Iteration $i logged to $LOG_FILE"
    echo ""
done

echo "All 50 iterations completed!"
