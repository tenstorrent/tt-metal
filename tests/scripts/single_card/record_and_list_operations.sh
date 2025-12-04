#!/bin/bash
# Script to record TTNN operations during model execution and extract them from the SQLite database
#
# Usage:
#   ./record_and_list_operations.sh <report_name> "<test_command>"
#
# Example:
#   ./record_and_list_operations.sh resnet_ops "pytest models/demos/wormhole/resnet50/demo/demo.py"
#
# This script:
#   1. Sets TTNN_CONFIG_OVERRIDES to enable operation logging
#   2. Runs the specified test command
#   3. Finds the generated db.sqlite file
#   4. Extracts and lists all operations from the database
#
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
echo "Recording TTNN Operations"
echo "=========================================="
echo "Report name: $REPORT_NAME"
echo "Test command: $TEST_CMD"
echo ""

# Set TTNN_CONFIG_OVERRIDES to enable operation logging
export TTNN_CONFIG_OVERRIDES='{
    "enable_fast_runtime_mode": false,
    "enable_logging": true,
    "report_name": "'"$REPORT_NAME"'",
    "enable_graph_report": false,
    "enable_detailed_buffer_report": true,
    "enable_detailed_tensor_report": false,
    "enable_comparison_mode": false
}'

echo "TTNN_CONFIG_OVERRIDES set to:"
echo "$TTNN_CONFIG_OVERRIDES"
echo ""

# Run the test command
echo "=========================================="
echo "Running test command..."
echo "=========================================="
eval "$TEST_CMD"
TEST_EXIT_CODE=$?

echo ""
echo "Test command finished with exit code: $TEST_EXIT_CODE"
echo ""

# Find the generated db.sqlite file
# Reports are generated at: generated/ttnn/reports/<report_name_hash>/db.sqlite
REPORTS_DIR="${TT_METAL_HOME:-$(pwd)}/generated/ttnn/reports"

echo "=========================================="
echo "Searching for SQLite database..."
echo "=========================================="
echo "Looking in: $REPORTS_DIR"

if [[ ! -d "$REPORTS_DIR" ]]; then
    echo "Error: Reports directory does not exist: $REPORTS_DIR"
    exit 1
fi

# Find the most recently modified db.sqlite file
DB_FILE=$(find "$REPORTS_DIR" -name "db.sqlite" -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)

if [[ -z "$DB_FILE" ]]; then
    # Fallback for macOS (which doesn't support -printf)
    DB_FILE=$(find "$REPORTS_DIR" -name "db.sqlite" -type f -exec stat -f "%m %N" {} \; 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
fi

if [[ -z "$DB_FILE" ]]; then
    echo "Error: No db.sqlite file found in $REPORTS_DIR"
    echo "Contents of reports directory:"
    ls -la "$REPORTS_DIR" 2>/dev/null || echo "(directory is empty or doesn't exist)"
    exit 1
fi

echo "Found database: $DB_FILE"
echo ""

# Get the report directory (parent of db.sqlite)
REPORT_DIR=$(dirname "$DB_FILE")
echo "Report directory: $REPORT_DIR"

# Check if config.json exists alongside db.sqlite
if [[ -f "$REPORT_DIR/config.json" ]]; then
    echo "Config file found: $REPORT_DIR/config.json"
else
    echo "Warning: config.json not found in report directory"
fi
echo ""

# Extract operations from the SQLite database
echo "=========================================="
echo "Extracting operations from database..."
echo "=========================================="

# Check if sqlite3 is available
if ! command -v sqlite3 &> /dev/null; then
    echo "Error: sqlite3 command not found. Please install sqlite3."
    exit 1
fi

# Get total count of operations
OP_COUNT=$(sqlite3 "$DB_FILE" "SELECT COUNT(*) FROM operations;")
echo "Total operations recorded: $OP_COUNT"
echo ""

if [[ "$OP_COUNT" -eq 0 ]]; then
    echo "Warning: No operations were recorded in the database."
    exit 0
fi

# List all operations with their details
echo "=========================================="
echo "Operations List (ID | Name | Duration)"
echo "=========================================="
sqlite3 -header -column "$DB_FILE" "SELECT operation_id, name, duration FROM operations ORDER BY operation_id;"

echo ""
echo "=========================================="
echo "Operations Summary by Name"
echo "=========================================="
sqlite3 -header -column "$DB_FILE" "
SELECT 
    name,
    COUNT(*) as count,
    ROUND(SUM(duration), 4) as total_duration,
    ROUND(AVG(duration), 4) as avg_duration
FROM operations 
GROUP BY name 
ORDER BY count DESC;
"

echo ""
echo "=========================================="
echo "Unique Operations (sorted alphabetically)"
echo "=========================================="
sqlite3 "$DB_FILE" "SELECT DISTINCT name FROM operations ORDER BY name;"

# Save operations to a text file for later reference
OUTPUT_FILE="${REPORT_DIR}/operations_list.txt"
echo ""
echo "=========================================="
echo "Saving operations list to: $OUTPUT_FILE"
echo "=========================================="

{
    echo "TTNN Operations Report"
    echo "Report Name: $REPORT_NAME"
    echo "Generated: $(date)"
    echo "Database: $DB_FILE"
    echo ""
    echo "Total Operations: $OP_COUNT"
    echo ""
    echo "=========================================="
    echo "All Operations (ID | Name | Duration)"
    echo "=========================================="
    sqlite3 -header -column "$DB_FILE" "SELECT operation_id, name, duration FROM operations ORDER BY operation_id;"
    echo ""
    echo "=========================================="
    echo "Operations Summary by Name"
    echo "=========================================="
    sqlite3 -header -column "$DB_FILE" "
    SELECT 
        name,
        COUNT(*) as count,
        ROUND(SUM(duration), 4) as total_duration,
        ROUND(AVG(duration), 4) as avg_duration
    FROM operations 
    GROUP BY name 
    ORDER BY count DESC;
    "
    echo ""
    echo "=========================================="
    echo "Unique Operations"
    echo "=========================================="
    sqlite3 "$DB_FILE" "SELECT DISTINCT name FROM operations ORDER BY name;"
} > "$OUTPUT_FILE"

echo "Operations list saved to: $OUTPUT_FILE"
echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
echo "Report directory: $REPORT_DIR"
echo "Database file: $DB_FILE"
echo "Operations list: $OUTPUT_FILE"
echo ""

# Exit with the same code as the test command
exit $TEST_EXIT_CODE

