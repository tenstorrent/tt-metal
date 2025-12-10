#!/bin/bash
# Script to record TTNN operations during model execution using Tracy profiler
#
# Usage:
#   ./record_and_list_operations.sh <report_name> "<test_command>"
#
# Example:
#   ./record_and_list_operations.sh resnet_ops "pytest models/demos/wormhole/resnet50/demo/demo.py"
#
# This script:
#   1. Runs the test with Tracy profiling enabled
#   2. Finds the generated ops_perf_results CSV file
#   3. Extracts and lists all operations from the CSV
#
# Tracy captures operations WITHOUT requiring code changes or disabling trace mode.
# Reference: https://docs.tenstorrent.com/ttnn-visualizer/src/installing.html

set -e

REPORT_NAME="${1:-ttnn_ops_report}"
TEST_CMD="${2:-}"

if [[ -z "$TEST_CMD" ]]; then
    echo "Usage: $0 <report_name> <test_command>"
    echo "Example: $0 resnet_ops \"pytest models/demos/wormhole/resnet50/demo/demo.py\""
    exit 1
fi

echo "=========================================="
echo "Recording Operations via Tracy"
echo "=========================================="
echo "Report name: $REPORT_NAME"
echo "Test command: $TEST_CMD"
echo ""

# Set up output directory
PROFILER_OUTPUT_DIR="${TT_METAL_HOME:-$(pwd)}/generated/profiler"
mkdir -p "$PROFILER_OUTPUT_DIR"

# Run the test with Tracy profiling
# -r: Generate ops report
# -v: Verbose output
echo "=========================================="
echo "Running test with Tracy profiler..."
echo "=========================================="

# Modify pytest command to run under Tracy
# Key flags:
#   -r: Generate ops report
#   -p: Enable device profiler
#   --dump-device-data-mid-run: Flush device profiler buffer periodically (prevents overflow)
#   -m: Run the following as a Python module
if [[ "$TEST_CMD" == pytest* ]]; then
    # For pytest: use -m flag to run pytest as a module
    # Extract pytest args (everything after 'pytest ')
    PYTEST_ARGS="${TEST_CMD#pytest }"
    TRACY_CMD="python -m tracy -r -p --dump-device-data-mid-run -m pytest ${PYTEST_ARGS}"
else
    # Wrap non-pytest commands (python scripts)
    TRACY_CMD="python -m tracy -r -p --dump-device-data-mid-run -m '${TEST_CMD}'"
fi

echo "Tracy command: $TRACY_CMD"
echo ""

eval "$TRACY_CMD"
TEST_EXIT_CODE=$?

echo ""
echo "Test command finished with exit code: $TEST_EXIT_CODE"
echo ""

# Find the generated ops_perf_results CSV file
REPORTS_DIR="${TT_METAL_HOME:-$(pwd)}/generated/profiler/reports"

echo "=========================================="
echo "Searching for Tracy ops report in: $REPORTS_DIR ..."
echo "=========================================="

if [[ ! -d "$REPORTS_DIR" ]]; then
    echo "Error: Reports directory does not exist: $REPORTS_DIR"
    echo "Tracy profiling may have failed."
    exit 1
fi

# Find the most recent ops_perf_results CSV file
OPS_CSV=$(find "$REPORTS_DIR" -name "ops_perf_results_*.csv" -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)

if [[ -z "$OPS_CSV" ]]; then
    echo "Error: No ops_perf_results CSV file found in $REPORTS_DIR"
    echo "Contents of reports directory:"
    ls -la "$REPORTS_DIR" 2>/dev/null || echo "(directory is empty or doesn't exist)"
    exit 1
fi

echo "Found ops CSV: $OPS_CSV"
echo ""

# Get the report directory
REPORT_DIR=$(dirname "$OPS_CSV")
echo "Report directory: $REPORT_DIR"
echo ""

# Extract and display operations using Python
echo "=========================================="
echo "Extracting operations from Tracy report..."
echo "=========================================="

OUTPUT_FILE="${REPORT_DIR}/operations_list_${REPORT_NAME}.txt"

python3 << EOF
import csv
from collections import defaultdict
from datetime import datetime

ops_csv = "$OPS_CSV"
report_name = "$REPORT_NAME"
output_file = "$OUTPUT_FILE"

# Read the CSV file
ops_data = []
with open(ops_csv, 'r') as f:
    reader = csv.DictReader(f)
    ops_data = list(reader)

print(f"Total operations recorded: {len(ops_data)}")
print()

if len(ops_data) == 0:
    print("Warning: No operations were recorded.")
    exit(0)

# Aggregate by operation name (OP CODE contains the actual operation)
op_counts = defaultdict(int)

for op in ops_data:
    op_name = op.get("OP CODE", op.get("op_code", "unknown"))
    op_counts[op_name] += 1

# Save report
with open(output_file, 'w') as f:
    f.write("Operations Report (via Tracy Profiler)\n")
    f.write(f"Report Name: {report_name}\n")
    f.write(f"Source CSV: {ops_csv}\n")

    f.write("Unique Operations\n")
    f.write("=" * 60 + "\n")
    for op_name in sorted(op_counts.keys()):
        f.write(f"{op_name}\n")

print(f"Report saved to: {output_file}")
EOF

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
echo "Report directory: $REPORT_DIR"
echo "Operations CSV: $OPS_CSV"
echo "Operations list: $OUTPUT_FILE"
echo ""

# Exit with the same code as the test command
exit $TEST_EXIT_CODE
