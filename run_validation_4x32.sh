#!/bin/bash

# Script to run cluster validation commands for 50 iterations
# Each iteration's output is logged to a separate file

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
        tt-smi -glx_reset

        echo ""
        echo "Running cluster validation..."
        ./build_Release/tools/scaleout/run_cluster_validation --factory-descriptor-path /data/local-syseng-manual/4x4x32_fsd.textproto --send-traffic --num-iterations 10
        echo "Iteration $i completed at $(date)"
        echo "=========================================="
    } 2>&1 | tee "$LOG_FILE"

    echo "Iteration $i logged to $LOG_FILE"
    echo ""
done

echo "All 50 iterations completed!"
