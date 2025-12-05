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

# Export TT_DISABLE_TRACE if set (disables trace capture for ops recording compatibility)
if [[ -n "$TT_DISABLE_TRACE" ]]; then
    export TT_DISABLE_TRACE
    echo "TT_DISABLE_TRACE=$TT_DISABLE_TRACE (trace capture disabled)"
fi

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

# Extract operations from the SQLite database using Python (sqlite3 module is built-in)
echo "=========================================="
echo "Extracting operations from database..."
echo "=========================================="

OUTPUT_FILE="${REPORT_DIR}/operations_list.txt"

python3 << EOF
import sqlite3
from datetime import datetime

db_file = "$DB_FILE"
report_name = "$REPORT_NAME"
output_file = "$OUTPUT_FILE"

conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Get total count
cursor.execute("SELECT COUNT(*) FROM operations")
op_count = cursor.fetchone()[0]
print(f"Total operations recorded: {op_count}")
print()

if op_count == 0:
    print("Warning: No operations were recorded in the database.")
    conn.close()
    exit(0)

# List all operations
print("=" * 50)
print("Operations List (ID | Name | Duration)")
print("=" * 50)
cursor.execute("SELECT operation_id, name, duration FROM operations ORDER BY operation_id")
print(f"{'ID':<10} {'Name':<50} {'Duration':<15}")
print("-" * 75)
for row in cursor.fetchall():
    op_id, name, duration = row
    duration_str = f"{duration:.6f}" if duration else "N/A"
    print(f"{op_id:<10} {name:<50} {duration_str:<15}")

# Operations summary
print()
print("=" * 50)
print("Operations Summary by Name")
print("=" * 50)
cursor.execute("""
    SELECT 
        name,
        COUNT(*) as count,
        ROUND(SUM(duration), 4) as total_duration,
        ROUND(AVG(duration), 4) as avg_duration
    FROM operations 
    GROUP BY name 
    ORDER BY count DESC
""")
print(f"{'Name':<50} {'Count':<10} {'Total':<15} {'Avg':<15}")
print("-" * 90)
for row in cursor.fetchall():
    name, count, total_dur, avg_dur = row
    total_str = f"{total_dur:.4f}" if total_dur else "N/A"
    avg_str = f"{avg_dur:.4f}" if avg_dur else "N/A"
    print(f"{name:<50} {count:<10} {total_str:<15} {avg_str:<15}")

# Unique operations
print()
print("=" * 50)
print("Unique Operations (sorted alphabetically)")
print("=" * 50)
cursor.execute("SELECT DISTINCT name FROM operations ORDER BY name")
for row in cursor.fetchall():
    print(row[0])

# Save to file
print()
print("=" * 50)
print(f"Saving operations list to: {output_file}")
print("=" * 50)

with open(output_file, 'w') as f:
    f.write("TTNN Operations Report\n")
    f.write(f"Report Name: {report_name}\n")
    f.write(f"Generated: {datetime.now()}\n")
    f.write(f"Database: {db_file}\n")
    f.write(f"\nTotal Operations: {op_count}\n\n")
    
    f.write("=" * 50 + "\n")
    f.write("All Operations (ID | Name | Duration)\n")
    f.write("=" * 50 + "\n")
    cursor.execute("SELECT operation_id, name, duration FROM operations ORDER BY operation_id")
    for row in cursor.fetchall():
        op_id, name, duration = row
        duration_str = f"{duration:.6f}" if duration else "N/A"
        f.write(f"{op_id}\t{name}\t{duration_str}\n")
    
    f.write("\n" + "=" * 50 + "\n")
    f.write("Operations Summary by Name\n")
    f.write("=" * 50 + "\n")
    cursor.execute("""
        SELECT name, COUNT(*) as count, ROUND(SUM(duration), 4), ROUND(AVG(duration), 4)
        FROM operations GROUP BY name ORDER BY count DESC
    """)
    for row in cursor.fetchall():
        name, count, total_dur, avg_dur = row
        f.write(f"{name}\t{count}\t{total_dur}\t{avg_dur}\n")
    
    f.write("\n" + "=" * 50 + "\n")
    f.write("Unique Operations\n")
    f.write("=" * 50 + "\n")
    cursor.execute("SELECT DISTINCT name FROM operations ORDER BY name")
    for row in cursor.fetchall():
        f.write(f"{row[0]}\n")

conn.close()
print(f"Operations list saved to: {output_file}")
EOF

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

