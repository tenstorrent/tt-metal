#!/usr/bin/env bash

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Check if CHIP_ARCH is set to wormhole or blackhole
if [[ "$CHIP_ARCH" = "wormhole" ]]; then
    /home/software/syseng/wh/tt-smi -wr 0
elif [[ "$CHIP_ARCH" = "blackhole" ]]; then
    tt-smi -r 0
else
    echo "No architecture detected"
fi

cd .. && make clean && cd python_tests
rm -rf *.log

# Function to display usage instructions
usage() {
    echo "Usage: $0 --repeat <number_of_repeats> --test <test_name> [--log <log_file>]"
    echo "       $0 --all "
    exit 1
}

# Default values
log_file="test_results.log"  # Default log file
repeat_count=1
test_name=""
all_tests=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --repeat)
            if [[ -n "$2" && "$2" =~ ^[0-9]+$ ]]; then
                repeat_count="$2"
                shift 2
            else
                echo "Error: --repeat requires a numeric argument."
                usage
            fi
            ;;
        --test)
            if [[ -n "$2" ]]; then
                test_name="$2"
                shift 2
            else
                echo "Error: --test requires an argument."
                usage
            fi
            ;;
        --all)
            all_tests=true
            shift
            ;;
        --log)
            if [[ -n "$2" ]]; then
                log_file="$2"
                shift 2
            else
                echo "Error: --log requires an argument."
                usage
            fi
            ;;
        *)
            usage
            ;;
    esac
done


# Ensure both parameters (--repeat and --test) are provided if not running all tests
if [[ -z "$repeat_count" || -z "$test_name" ]]; then
    usage
fi

# Run the test for the specified number of iterations
for i in $(seq 1 "$repeat_count"); do
    if [[ -n "$log_file" ]]; then
        echo "Running test: $test_name (Iteration $i)" | tee -a "$log_file"
        pytest --color=yes -v "$test_name" | tee -a "$log_file" 2>&1
    else
        echo "Running test: $test_name (Iteration $i)"
        pytest --tb=short --color=yes -v "$test_name"
    fi

    result=$?

    if [ [$result -eq 0] ]; then
        ((pass_count++))
    else
        ((fail_count++))
    fi
done

# Print summary
echo "Summary:"
echo "Passed: $pass_count"
echo "Failed: $fail_count"
