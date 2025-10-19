#!/bin/bash
export TT_METAL_HOME=$PWD

echo "Starting cluster validation loop - 10 iterations"
echo "================================================"

for i in {1..10}; do
    echo ""
    echo "=== Iteration $i/100 ==="
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"

    # Reset using tt-smi
    echo "Performing GLX reset..."
    tt-smi -glx_reset > log.txt

    # Run cluster validation
    echo "Running cluster validation..."

    ./build/tools/scaleout/run_cluster_validation  --cabling-descriptor-path tools/tests/scaleout/cabling_descriptors/4_wh_galaxy_y_torus_a2a.textproto --deployment-descriptor-path tools/tests/scaleout/deployment_descriptors/4_glx_wh_a2a.textproto --print-connectivity --send-traffic

     if [ $? -eq 0 ]; then
        echo "Iteration $i completed successfully"
    else
        echo "Iteration $i failed with exit code $?"
        echo "Continuing to next iteration..."
    fi

    echo "----------------------------------------"
done

echo ""
echo "================================================"
echo "All 100 iterations completed"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
