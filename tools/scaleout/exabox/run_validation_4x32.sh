#!/bin/bash

# Script to run cluster validation commands for 50 iterations
# Each iteration's output is logged to a separate file
#
# Usage: ./run_validation_4x32.sh <comma-separated-host-list>
# Example: ./run_validation_4x32.sh bh-glx-c01u02,bh-glx-c01u08,bh-glx-c02u02,bh-glx-c02u08

# Check if host list is provided
if [ $# -eq 0 ]; then
    echo "Error: No host list provided"
    echo "Usage: $0 <comma-separated-host-list> <docker-image>"
    echo "Example: $0 bh-glx-c01u02,bh-glx-c01u08,bh-glx-c02u02,bh-glx-c02u08 ghcr.io/tenstorrent/tt-metal/upstream-tests-wh-6u:latest"
    exit 1
fi

# Get host list from command line argument
HOSTS="$1"
DOCKER_IMAGE="$2"
echo "Using hosts: $HOSTS"
echo "Using docker image: $DOCKER_IMAGE"
echo ""

# Create output directory if it doesn't exist
mkdir -p validation_output

for i in {1..50}; do
    echo "Starting iteration $i of 50..."

    LOG_FILE="validation_output/cluster_validation_iteration_${i}.log"

    {
        echo "=========================================="
        echo "Iteration: $i"
        echo "Timestamp: $(date)"
        echo "=========================================="
        echo ""

        echo "Running tt-smi -glx_reset..."
        mpirun --host $HOSTS --mca btl_tcp_if_exclude docker0,lo tt-smi -glx_reset

        echo ""
        echo "Running cluster validation..."
        ./tools/scaleout/exabox/mpi-docker --image $DOCKER_IMAGE --empty-entrypoint --host $HOSTS ./build/tools/scaleout/run_cluster_validation --factory-descriptor-path /data/scaleout_configs/4xBH_4x32_intrapod/fsd.textproto --send-traffic --num-iterations 10
        echo "Iteration $i completed at $(date)"
        echo "=========================================="
    } 2>&1 | tee "$LOG_FILE"

    echo "Iteration $i logged to $LOG_FILE"
    echo ""
done

echo "All 50 iterations completed!"
