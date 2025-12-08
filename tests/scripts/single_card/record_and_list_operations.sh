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
echo "Recording TTNN Operations via Tracy"
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
if [[ "$TEST_CMD" == pytest* ]]; then
    # For pytest: use -m flag to run pytest as a module
    # Extract pytest args (everything after 'pytest ')
    PYTEST_ARGS="${TEST_CMD#pytest }"
    TRACY_CMD="python -m tracy -r -m pytest ${PYTEST_ARGS}"
else
    # Wrap non-pytest commands (python scripts)
    TRACY_CMD="python -m tracy -r -m '${TEST_CMD}'"
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
echo "Searching for Tracy ops report..."
echo "=========================================="
echo "Looking in: $REPORTS_DIR"

if [[ ! -d "$REPORTS_DIR" ]]; then
    echo "Error: Reports directory does not exist: $REPORTS_DIR"
    echo "Tracy profiling may have failed."
    exit 1
fi

# Find the most recent ops_perf_results CSV file
OPS_CSV=$(find "$REPORTS_DIR" -name "ops_perf_results_*.csv" -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)

if [[ -z "$OPS_CSV" ]]; then
    # Fallback for macOS
    OPS_CSV=$(find "$REPORTS_DIR" -name "ops_perf_results_*.csv" -type f -exec stat -f "%m %N" {} \; 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
fi

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

# Aggregate by operation type
op_summary = defaultdict(lambda: {"count": 0, "total_duration_ns": 0, "calls": []})

for op in ops_data:
    op_type = op.get("OP TYPE", op.get("op_type", "unknown"))
    op_code = op.get("OP CODE", op.get("op_code", ""))
    
    # Get duration - try different possible column names
    duration = 0
    for col in ["DEVICE FW DURATION [ns]", "device_fw_duration", "KERNEL DURATION [ns]", "kernel_duration"]:
        if col in op and op[col]:
            try:
                duration = float(op[col])
                break
            except:
                pass
    
    op_summary[op_type]["count"] += 1
    op_summary[op_type]["total_duration_ns"] += duration
    op_summary[op_type]["calls"].append(op)

# Print operations list
print("=" * 70)
print("Operations Summary by Type")
print("=" * 70)
print(f"{'Operation':<45} {'Count':<10} {'Total Duration (ms)':<20}")
print("-" * 75)

for op_type, data in sorted(op_summary.items(), key=lambda x: x[1]["count"], reverse=True):
    duration_ms = data["total_duration_ns"] / 1_000_000
    print(f"{op_type:<45} {data['count']:<10} {duration_ms:<20.4f}")

print()
print("=" * 70)
print("Unique Operations (sorted alphabetically)")
print("=" * 70)
for op_type in sorted(op_summary.keys()):
    print(op_type)

# Save detailed report
print()
print("=" * 70)
print(f"Saving detailed report to: {output_file}")
print("=" * 70)

with open(output_file, 'w') as f:
    f.write("TTNN Operations Report (via Tracy Profiler)\n")
    f.write(f"Report Name: {report_name}\n")
    f.write(f"Generated: {datetime.now()}\n")
    f.write(f"Source CSV: {ops_csv}\n")
    f.write(f"\nTotal Operations: {len(ops_data)}\n")
    f.write(f"Unique Operation Types: {len(op_summary)}\n\n")
    
    f.write("=" * 70 + "\n")
    f.write("Operations Summary by Type\n")
    f.write("=" * 70 + "\n")
    f.write(f"{'Operation':<45} {'Count':<10} {'Total Duration (ms)':<20}\n")
    f.write("-" * 75 + "\n")
    
    for op_type, data in sorted(op_summary.items(), key=lambda x: x[1]["count"], reverse=True):
        duration_ms = data["total_duration_ns"] / 1_000_000
        f.write(f"{op_type:<45} {data['count']:<10} {duration_ms:<20.4f}\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write("Unique Operations\n")
    f.write("=" * 70 + "\n")
    for op_type in sorted(op_summary.keys()):
        f.write(f"{op_type}\n")
    
    # Also save raw data reference
    f.write("\n" + "=" * 70 + "\n")
    f.write("Full CSV Data Available At\n")
    f.write("=" * 70 + "\n")
    f.write(f"{ops_csv}\n")

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
