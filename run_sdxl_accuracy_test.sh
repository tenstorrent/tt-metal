#!/bin/bash

# SDXL Accuracy Test Runner Script
# Runs test_accuracy_sdxl with specific parameters and detailed logging

# Set script variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_FILE="models/experimental/stable_diffusion_xl_base/tests/test_sdxl_accuracy.py"
LOG_FILE="sdxl_accuracy_test_$(date +%Y%m%d_%H%M%S).log"
TEST_FILTER="with_trace and host_vae"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to log with timestamp
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to log colored output
log_colored() {
    local color=$1
    local message=$2
    echo -e "${color}[$(date '+%Y-%m-%d %H:%M:%S')] ${message}${NC}" | tee -a "$LOG_FILE"
}

echo "=================================="
log_colored "$BLUE" "SDXL Accuracy Test Runner Starting"
echo "=================================="

# Check if test file exists
if [[ ! -f "$TEST_FILE" ]]; then
    log_colored "$RED" "ERROR: Test file not found: $TEST_FILE"
    exit 1
fi

log_with_timestamp "Test file: $TEST_FILE"
log_with_timestamp "Test filter: $TEST_FILTER"
log_with_timestamp "Log file: $LOG_FILE"
log_with_timestamp "Working directory: $(pwd)"

# Set environment variable
export TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="7,7"
log_with_timestamp "Environment variable set: TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=$TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE"

# Print system info
log_with_timestamp "Python version: $(python --version 2>&1)"
log_with_timestamp "Pytest version: $(python -m pytest --version 2>&1 | head -1)"

log_colored "$YELLOW" "Starting pytest execution..."
echo "=================================="

# Run pytest with verbose output and capture both stdout and stderr
{
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] TEST EXECUTION STARTED"

    # Use pytest with detailed output
    python -m pytest \
        "$TEST_FILE::test_accuracy_sdxl" \
        -k "$TEST_FILTER" \
        -v \
        -s \
        --tb=short \
        --capture=no \
        --log-cli-level=INFO \
        --log-cli-format='[%(asctime)s] %(levelname)s: %(message)s' \
        --log-cli-date-format='%Y-%m-%d %H:%M:%S' 2>&1 | while IFS= read -r line; do

        # Check for test start patterns
        if [[ "$line" =~ test_accuracy_sdxl.*PASSED|test_accuracy_sdxl.*FAILED|test_accuracy_sdxl.*SKIPPED ]]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] TEST FINISHED: $line"
        elif [[ "$line" =~ test_accuracy_sdxl ]]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] TEST PROGRESS: $line"
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"
        fi
    done

    # Capture the exit code
    TEST_EXIT_CODE=${PIPESTATUS[0]}
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] TEST EXECUTION FINISHED WITH EXIT CODE: $TEST_EXIT_CODE"

} >> "$LOG_FILE" 2>&1

# Get the actual exit code
TEST_EXIT_CODE=$?

echo "=================================="
if [[ $TEST_EXIT_CODE -eq 0 ]]; then
    log_colored "$GREEN" "✅ Tests completed successfully!"
else
    log_colored "$RED" "❌ Tests failed with exit code: $TEST_EXIT_CODE"
fi

echo "=================================="
log_with_timestamp "Test execution completed"
log_with_timestamp "Full log saved to: $LOG_FILE"
log_with_timestamp "You can view the log with: tail -f $LOG_FILE"

# Show log file location
echo ""
echo "Log file location: $(realpath "$LOG_FILE")"
echo ""
echo "To monitor progress in real-time, run:"
echo "tail -f $LOG_FILE"

exit $TEST_EXIT_CODE
