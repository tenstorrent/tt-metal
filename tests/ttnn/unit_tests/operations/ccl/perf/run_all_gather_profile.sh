#!/bin/sh
MODULE_DIR="tests/ttnn/unit_tests/operations/ccl/perf"

# Defaults
DEBUG=false
TARGET="n300"

set -e

# Function to display help
show_help() {
    echo "Usage: ./tests/ttnn/unit_tests/operations/ccl/perf/run_profile.sh [OPTIONS]"
    echo
    echo "Options:"
    echo "  -d, --debug        Enable debug mode to show real-time output."
    echo "  -t, --target       Specify the target configuration (t3000 or n300 or tg). Default is n300."
    echo "  -h, --help         Display this help message."
    echo
    echo "Example:"
    echo "  ./tests/ttnn/unit_tests/operations/ccl/perf/run_profile.sh --debug --target n300"
    echo "  ./tests/ttnn/unit_tests/operations/ccl/perf/run_profile.sh -h"
}

# Parse command-line arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --debug|-d)
            DEBUG=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        --target|-t)
            # Ensure there is an argument following the target flag
            if [ -z "$2" ]; then
                echo "Error: No target specified after $1."
                show_help
                exit 1
            fi

            TARGET="$2"  # Set the target configuration
            shift 2

            # Validate the target value
            if [ "$TARGET" != "t3000" ] && [ "$TARGET" != "tg" ] && [ "$TARGET" != "n300" ]; then
                echo "Error: Invalid target configuration: $TARGET. Must be 't3000' or 'n300' or 'tg'."
                exit 1
            fi
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Function to run the profiling command and extract the CSV path
run_profile_and_extract_csv() {
    command="./tt_metal/tools/profiler/profile_this.py -n all_gather_$TARGET -c 'pytest tests/ttnn/unit_tests/operations/ccl/perf/test_ccl_perf.py::test_all_gather_on_$TARGET'"

    if [ "$DEBUG" = true ]; then
        echo "Running profiling command for target $TARGET in debug mode..."
        full_output=$(eval $command 2>&1 | tee /dev/tty)
    else
        echo "Running profiling command for target $TARGET..."
        full_output=$(eval $command 2>&1)
    fi

    # Extract the CSV path
    csv_path=$(echo "$full_output" | grep -oE 'OPs csv generated at: (.+\.csv)' | sed -E 's/OPs csv generated at: //')

    if [ -n "$csv_path" ]; then
        echo "CSV path found: $csv_path"
        echo "Generating performance report..."

        tmp_file="/tmp/perf_report_output.log"
        PYTHONPATH="$MODULE_DIR" python3 -c "
import sys
import pandas as pd
from perf_csv import perf_report
from tabulate import tabulate

try:
    # Generate the report and convert it to a DataFrame
    average_df, utilization = perf_report('$csv_path')
    # Print the DataFrame in a pretty table format
    print('Min - Avg - Max by Common Runs:')
    print(tabulate(average_df, headers='keys', tablefmt='pretty'))
    print('Device Utilization:')
    print(tabulate(utilization, headers='keys', tablefmt='pretty'))
except Exception as e:
    print(f'Error in performance report generation: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1 | tee "$tmp_file"

        if grep -q "Error in performance report generation" "$tmp_file"; then
            echo "Error: Performance report generation failed."
            exit 1
        fi

    else
        echo "CSV path not found in the command output."
        exit 1
    fi
}

# Run the function
run_profile_and_extract_csv
