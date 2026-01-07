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
        mpirun --host bh-glx-c01u02,bh-glx-c01u08,bh-glx-c02u02,bh-glx-c02u08 --mca btl_tcp_if_include ens5f0np0 tt-smi -glx_reset

        echo ""
        echo "Running cluster validation..."
        mpirun --host bh-glx-c01u02,bh-glx-c01u08,bh-glx-c02u02,bh-glx-c02u08 --mca btl_tcp_if_include ens5f0np0 --tag-output ./build_Release/tools/scaleout/run_cluster_validation --factory-descriptor-path 5x8x16_fsd.textproto --send-traffic --num-iterations 1
	source python_env/bin/activate
	export TT_METAL_HOME=/home/local-syseng/tt-metal
	tt-run --rank-binding tests/tt_metal/distributed/config/quad_galaxy_rank_bindings.yaml --mpi-args "--host bh-glx-c01u02,bh-glx-c01u08,bh-glx-c02u02,bh-glx-c02u08 --mca btl_tcp_if_include ens5f0np0 --tag-output" ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_dual_galaxy_fabric_2d_sanity.yaml
	echo ""
        echo "Iteration $i completed at $(date)"
        echo "=========================================="
    } 2>&1 | tee "$LOG_FILE"

    echo "Iteration $i logged to $LOG_FILE"
    echo ""
done

echo "All 50 iterations completed!"
