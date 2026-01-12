#!/bin/bash
mkdir -p fabric_test_logs
source python_env/bin/activate
export TT_METAL_HOME=/home/local-syseng/tt-metal
HOSTS="$1"
LOG_FILE="fabric_test_logs/fabric_tests_$(date +%Y%m%d_%H%M%S).log"

echo "Running fabric tests..."
echo "Using hosts: $HOSTS"
echo "Logging to: $LOG_FILE"
echo ""

{
    echo "=========================================="
    echo "Running fabric tests..."
    echo "Using hosts: $HOSTS"
    echo "Logging to: $LOG_FILE"
    echo "=========================================="
    echo ""

    tt-run --rank-binding tests/tt_metal/distributed/config/quad_galaxy_rank_bindings.yaml --mpi-args "--host $HOSTS --mca btl_tcp_if_include ens5f0np0 --tag-output" ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_8x16_stability.yaml
} 2>&1 | tee "$LOG_FILE"

echo "Fabric tests completed!"
echo "Logs saved to: $LOG_FILE"
