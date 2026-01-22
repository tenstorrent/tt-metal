#!/bin/bash
# Measure telemetry impact using existing pytest tests that DO work
# This bypasses the CreateDevice bug by using pytest's working fixtures

set -e

TELEMETRY_BIN="/localdev/kkfernandez/tt-telemetry/build_Release/bin/tt_telemetry_server"
FSD_PATH="/localdev/kkfernandez/fsd.textproto"
TT_METAL_HOME="/localdev/kkfernandez/tt-metal"
OUTPUT_DIR="/tmp/telemetry_test_results_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_DIR"

echo "================================================================================"
echo "TELEMETRY IMPACT MEASUREMENT - Using Existing Pytest Tests"
echo "================================================================================"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""

# Find a working test
cd "$TT_METAL_HOME"

# Test to use - pick a simple one that works
# You can change this to any test that runs successfully on your system
TEST_SINGLE="tests/ttnn/integration_tests/test_add.py::test_add -k test_add"
TEST_MULTI="tests/ttnn/unit_tests/operations/ccl/test_all_gather_matmul_1d_tensor_nightly.py::test_all_gather_on_t3000_freq_mesh -k nightly"

# Pick whichever test exists and works
if pytest "$TEST_SINGLE" --collect-only > /dev/null 2>&1; then
    SINGLE_TEST="$TEST_SINGLE"
    echo "✓ Using single-device test: $SINGLE_TEST"
elif pytest "tests/ttnn/unit_tests/test_multi_device.py" --collect-only > /dev/null 2>&1; then
    SINGLE_TEST="tests/ttnn/unit_tests/test_multi_device.py::test_device_mesh"
    echo "✓ Using test: $SINGLE_TEST"
else
    # Fallback: just run any test in operations
    SINGLE_TEST=$(find tests/ttnn/unit_tests/operations -name "test_*.py" | head -1)"::test"
    echo "Using fallback test: $SINGLE_TEST"
fi

N_RUNS=5
POLLING_FREQS=("5s" "1s" "100ms" "10ms")

# Function to run test and measure time
run_test_iteration() {
    local telemetry_enabled=$1
    local polling_interval=$2
    local iteration=$3

    if [ "$telemetry_enabled" = "true" ]; then
        # Start telemetry
        $TELEMETRY_BIN \
            --polling-interval "$polling_interval" \
            --port 7070 \
            --fsd "$FSD_PATH" \
            > /dev/null 2>&1 &
        local telem_pid=$!
        sleep 3
    fi

    # Run test and measure time
    local start=$(date +%s.%N)
    timeout 120 pytest "$SINGLE_TEST" -v --tb=no > /dev/null 2>&1
    local exit_code=$?
    local end=$(date +%s.%N)

    if [ "$telemetry_enabled" = "true" ]; then
        kill $telem_pid 2>/dev/null || true
        wait $telem_pid 2>/dev/null || true
        sleep 1
    fi

    if [ $exit_code -ne 0 ]; then
        echo "0"  # Return 0 to indicate failure
        return 1
    fi

    # Calculate duration in milliseconds
    local duration=$(echo "($end - $start) * 1000" | bc)
    echo "$duration"
}

# Arrays for results
declare -A results

echo "Running $N_RUNS iterations per configuration..."
echo ""

# Baseline (no telemetry)
echo "=== BASELINE (No Telemetry) ==="
results["baseline"]=""
for i in $(seq 1 $N_RUNS); do
    time=$(run_test_iteration false "" $i)
    if [ "$time" != "0" ]; then
        results["baseline"]+="$time "
        printf "  Run %d: %8.2f ms\n" $i $time
    else
        echo "  Run $i: FAILED"
    fi
done
echo ""

# Test with different polling frequencies
for freq in "${POLLING_FREQS[@]}"; do
    echo "=== TELEMETRY: $freq polling ==="
    results["$freq"]=""
    for i in $(seq 1 $N_RUNS); do
        time=$(run_test_iteration true "$freq" $i)
        if [ "$time" != "0" ]; then
            results["$freq"]+="$time "
            printf "  Run %d: %8.2f ms\n" $i $time
        else
            echo "  Run $i: FAILED"
        fi
    done
    echo ""
done

# Calculate statistics and save results
python3 << PYEOF
import json
import statistics

results = {}

# Parse bash array into Python
for config in ['baseline'] + [${POLLING_FREQS[@]}]:
    times_str = "${results[config]}" if config in ${!results[@]} else ""
    if times_str.strip():
        times = [float(x) for x in times_str.strip().split()]
        if times:
            results[config] = {
                'times': times,
                'mean_ms': statistics.mean(times),
                'stdev_ms': statistics.stdev(times) if len(times) > 1 else 0,
                'n_runs': len(times)
            }

# Calculate overhead
if 'baseline' in results:
    baseline_mean = results['baseline']['mean_ms']
    for config, data in results.items():
        if config != 'baseline':
            overhead = ((data['mean_ms'] - baseline_mean) / baseline_mean) * 100
            data['overhead_pct'] = overhead

# Save to JSON
with open('$OUTPUT_DIR/results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print summary
print("="*80)
print("SUMMARY")
print("="*80)
print("")
print(f"{'Configuration':<20} {'Mean':<12} {'StdDev':<10} {'Overhead':<10}")
print("-"*80)

if 'baseline' in results:
    r = results['baseline']
    print(f"{'Baseline':<20} {r['mean_ms']:>8.2f} ms  {r['stdev_ms']:>6.2f} ms  {'---':<10}")

for config in [${POLLING_FREQS[@]}]:
    if config in results:
        r = results[config]
        oh = r.get('overhead_pct', 0)
        print(f"{config + ' polling':<20} {r['mean_ms']:>8.2f} ms  {r['stdev_ms']:>6.2f} ms  {oh:>+8.2f}%")

print("")
print(f"Results saved to: $OUTPUT_DIR/results.json")
PYEOF

echo ""
echo "================================================================================"
echo "Test complete!"
echo "================================================================================"
