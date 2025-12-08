#!/bin/bash

# Wrapper script to run SDXL accuracy test n times with specific configuration, which represents a stress test.
# Number of runs is picked such that the test runs for 24 hours.
# This runs the test with device encoders, trace enabled, device VAE, and no CFG parallel

LOG_FILE="sdxl_stress_test_results_$(date +%Y%m%d_%H%M%S).log"
NUM_RUNS=50
TEST_COMMAND="TT_SDXL_SKIP_CHECK_AND_SAVE=1 TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=7,7 TT_MM_THROTTLE_PERF=5 pytest models/experimental/stable_diffusion_xl_base/tests/test_sdxl_accuracy.py::test_accuracy_sdxl -k \"device_encoders and with_trace and device_vae and no_cfg_parallel and no_refiner\" --num-prompts=5000 -v -s"

echo "Running SDXL accuracy test with configuration:"
echo "- device_encoders: enabled"
echo "- with_trace: enabled"
echo "- device_vae: enabled"
echo "- no_cfg_parallel: enabled"
echo "- no_refiner: enabled"
echo "- Number of runs: ${NUM_RUNS}"
echo "- Log file: ${LOG_FILE}"
echo ""

# Change to the project root directory
# Get the directory where this script is located and navigate to tt-metal root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"
cd "$PROJECT_ROOT"

# Initialize log file
echo "SDXL Stress Test Results - $(date)" > "${LOG_FILE}"
echo "Configuration: device_encoders and with_trace and device_vae and no_cfg_parallel" >> "${LOG_FILE}"
echo "Number of runs: ${NUM_RUNS}" >> "${LOG_FILE}"
echo "========================================" >> "${LOG_FILE}"

# Run the test multiple times
for i in $(seq 1 ${NUM_RUNS}); do
    echo "Starting run ${i}/${NUM_RUNS}..." | tee -a "${LOG_FILE}"
    echo "Run ${i} started at: $(date)" >> "${LOG_FILE}"
    echo "----------------------------------------" >> "${LOG_FILE}"

    # Run the test and capture both stdout and stderr
    eval ${TEST_COMMAND} 2>&1 | tee -a "${LOG_FILE}"

    # Capture exit code
    exit_code=${PIPESTATUS[0]}
    echo "Run ${i} completed with exit code: ${exit_code}" >> "${LOG_FILE}"
    echo "Run ${i} ended at: $(date)" >> "${LOG_FILE}"
    echo "========================================" >> "${LOG_FILE}"

    if [ ${i} -lt ${NUM_RUNS} ]; then
        echo "Waiting 10 seconds before next run..." | tee -a "${LOG_FILE}"
        sleep 10
    fi
done

echo "All ${NUM_RUNS} test runs completed. Results saved to: ${LOG_FILE}"
