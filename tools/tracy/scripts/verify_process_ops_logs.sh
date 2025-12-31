#!/bin/bash
# Test script to compare current commit runtime analysis vs previous commit Python processing

set +e

# Activate virtual environment if it exists
if [ -d "python_env" ]; then
    source python_env/bin/activate
    echo "Activated virtual environment: python_env"
fi

# Use python3 if python is not available
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    PYTHON_CMD="python3"
fi

# Base tracy command parts
TRACY_BASE="$PYTHON_CMD -m tracy -v -r -p"
PYTEST_CMD="-m pytest models/demos/ttnn_resnet/tests/test_resnet50_performant.py::test_run_resnet50_trace_2cqs_inference[wormhole_b0-16-DataType.BFLOAT8_B-DataType.BFLOAT8_B-MathFidelity.LoFi-device_params0]"
BASE_DIR="test_runtime_analysis_outputs"

echo "=== Comparing Current Commit Runtime Analysis vs Previous Commit Python Processing ==="
echo "Base tracy command: $TRACY_BASE"
echo "Pytest command: $PYTEST_CMD"
echo "Output directory: $BASE_DIR"
echo ""

# Clean up previous test outputs
rm -rf "$BASE_DIR"
mkdir -p "$BASE_DIR"

# Get commit info
CURRENT_COMMIT=$(git rev-parse HEAD)
PREVIOUS_COMMIT=$(git rev-parse HEAD~1)

echo "Current commit: $CURRENT_COMMIT"
echo "Previous commit: $PREVIOUS_COMMIT"
echo ""

# Function to extract the latest CSV report path
get_latest_csv() {
    # Find the most recent dated folder in generated/profiler/reports/
    local latest_report=$(find generated/profiler/reports -type d -name "20*" 2>/dev/null | sort | tail -1)
    if [ -n "$latest_report" ]; then
        echo "$latest_report/ops_perf_results_$(basename $latest_report).csv"
    else
        echo ""
    fi
}

# Function to get the profiler artifacts directory
get_profiler_artifacts_dir() {
    # Check for TT_METAL_PROFILER_DIR environment variable
    if [ -n "$TT_METAL_PROFILER_DIR" ]; then
        echo "$TT_METAL_PROFILER_DIR"
        return
    fi

    # Default to generated/profiler
    if [ -d "generated/profiler" ]; then
        echo "generated/profiler"
    else
        echo ""
    fi
}

# Test scenarios
declare -a TEST_SCENARIOS=(
    "baseline:"
    "no_runtime_analysis:--no-runtime-analysis"
    "enable_sum_profiling:--enable-sum-profiling"
    "profile_dispatch_cores:--profile-dispatch-cores"
    "profiler_capture_perf_counters:--profiler-capture-perf-counters=all"
)

# Save current process_ops_logs.py
CURRENT_PROCESS_OPS_LOGS="tools/tracy/process_ops_logs.py"
BACKUP_PROCESS_OPS_LOGS="$BASE_DIR/process_ops_logs.py.current"
PREVIOUS_PROCESS_OPS_LOGS="$BASE_DIR/process_ops_logs.py.previous"

# Backup current version
cp "$CURRENT_PROCESS_OPS_LOGS" "$BACKUP_PROCESS_OPS_LOGS"

# Get previous commit's version
echo "=== Fetching previous commit's process_ops_logs.py ==="
git show "$PREVIOUS_COMMIT:$CURRENT_PROCESS_OPS_LOGS" > "$PREVIOUS_PROCESS_OPS_LOGS" 2>/dev/null
if [ ! -s "$PREVIOUS_PROCESS_OPS_LOGS" ]; then
    echo "Error: Could not retrieve previous commit's process_ops_logs.py"
    exit 1
fi
echo "✓ Retrieved previous commit's process_ops_logs.py"
echo ""

# Run tests on current commit
for scenario in "${TEST_SCENARIOS[@]}"; do
    IFS=':' read -r test_name tracy_options <<< "$scenario"

    echo "=== Testing: $test_name ==="

    # Step 1: Run test on current commit (generates CSV with runtime analysis)
    echo "Step 1: Running test on current commit..."
    $TRACY_BASE $tracy_options $PYTEST_CMD > "$BASE_DIR/${test_name}_current.log" 2>&1 || true

    LATEST_CSV=$(get_latest_csv)
    if [ -z "$LATEST_CSV" ] || [ ! -f "$LATEST_CSV" ]; then
        echo "✗ No CSV generated from current commit test"
        echo ""
        continue
    fi

    CURRENT_CSV="$BASE_DIR/${test_name}_current.csv"
    cp "$LATEST_CSV" "$CURRENT_CSV"
    echo "✓ Current commit CSV saved: $CURRENT_CSV"

    # Step 2: Get profiler artifacts directory for post-processing
    PROFILER_DIR=$(get_profiler_artifacts_dir)
    if [ -z "$PROFILER_DIR" ] || [ ! -d "$PROFILER_DIR" ]; then
        echo "✗ Could not find profiler artifacts directory for post-processing"
        echo ""
        continue
    fi
    echo "✓ Found profiler artifacts directory: $PROFILER_DIR"

    # Step 3: Use previous commit's process_ops_logs.py to regenerate CSV
    echo "Step 2: Regenerating CSV using previous commit's process_ops_logs.py..."
    cp "$PREVIOUS_PROCESS_OPS_LOGS" "$CURRENT_PROCESS_OPS_LOGS"

    # Run post-processing with previous commit's script
    # Use the same output folder so it processes the same logs
    $PYTHON_CMD tools/tracy/process_ops_logs.py -o "$PROFILER_DIR" --date > "$BASE_DIR/${test_name}_previous.log" 2>&1 || true

    # Restore current version
    cp "$BACKUP_PROCESS_OPS_LOGS" "$CURRENT_PROCESS_OPS_LOGS"

    LATEST_CSV=$(get_latest_csv)
    if [ -z "$LATEST_CSV" ] || [ ! -f "$LATEST_CSV" ]; then
        echo "✗ No CSV generated from previous commit's processing"
        echo ""
        continue
    fi

    PREVIOUS_CSV="$BASE_DIR/${test_name}_previous.csv"
    cp "$LATEST_CSV" "$PREVIOUS_CSV"
    echo "✓ Previous commit processing CSV saved: $PREVIOUS_CSV"

    # Step 4: Compare the two CSVs
    echo "Step 3: Comparing CSVs..."
    if [ ! -f "$CURRENT_CSV" ] || [ ! -f "$PREVIOUS_CSV" ]; then
        echo "✗ Missing CSV files for comparison"
        echo ""
        continue
    fi

    # Use compare_full_op_report.py
    if $PYTHON_CMD tools/tracy/compare_full_op_report.py "$CURRENT_CSV" "$PREVIOUS_CSV" --max-differences 50; then
        echo "✓ CSVs are identical (no regression)"
    else
        echo "✗ CSVs differ (potential regression detected)"
    fi
    echo ""
done

# Restore current version (safety check)
cp "$BACKUP_PROCESS_OPS_LOGS" "$CURRENT_PROCESS_OPS_LOGS"

echo "=== Test Summary ==="
echo "All outputs saved in: $BASE_DIR"
echo "Current commit: $CURRENT_COMMIT"
echo "Previous commit: $PREVIOUS_COMMIT"
echo ""
echo "CSV files:"
echo "  - *_current.csv: Generated by current commit's runtime analysis"
echo "  - *_previous.csv: Generated by previous commit's Python processing"
