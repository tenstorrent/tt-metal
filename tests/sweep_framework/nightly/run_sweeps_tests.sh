#!/bin/bash

# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# Script to run all sweep tests from vector export files
# This script iterates through all JSON files in vectors_export directory
# and runs the sweeps runner for each module

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_FRAMEWORK_DIR="$(dirname "$SCRIPT_DIR")"
VECTORS_EXPORT_DIR="$SWEEP_FRAMEWORK_DIR/vectors_export"
RUNNER_SCRIPT="$SWEEP_FRAMEWORK_DIR/sweeps_runner.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if vectors_export directory exists
if [ ! -d "$VECTORS_EXPORT_DIR" ]; then
    print_error "Vectors export directory not found: $VECTORS_EXPORT_DIR"
    print_error "Please run the parameter generator first with --dump-file flag"
    exit 1
fi

# Check if sweeps_runner.py exists
if [ ! -f "$RUNNER_SCRIPT" ]; then
    print_error "Sweeps runner script not found: $RUNNER_SCRIPT"
    exit 1
fi

# Change to the project root directory (assuming this is run from tt-metal root)
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SWEEP_FRAMEWORK_DIR")")")"
cd "$PROJECT_ROOT"

print_status "Changed to project root: $PROJECT_ROOT"

# Find all JSON files in vectors_export directory
json_files=($(find "$VECTORS_EXPORT_DIR" -name "*.json" -type f))

if [ ${#json_files[@]} -eq 0 ]; then
    print_warning "No JSON files found in $VECTORS_EXPORT_DIR"
    print_warning "Please run the parameter generator first with --dump-file flag"
    exit 0
fi

print_status "Found ${#json_files[@]} JSON files to process"

# Process each JSON file
for json_file in "${json_files[@]}"; do
    # Extract module name from filename (remove .json extension)
    filename=$(basename "$json_file")
    module_name="${filename%.json}"

    print_status "Processing module: $module_name"
    print_status "File: $json_file"

    # Run the sweeps runner for this module
    if python3 "$RUNNER_SCRIPT" --module-name "$module_name" --read-file "$json_file"; then
        print_success "Successfully completed sweeps for module: $module_name"
    else
        print_error "Failed to run sweeps for module: $module_name"
        # Continue with other modules even if one fails
    fi

    echo "----------------------------------------"
done

print_success "Completed processing all modules"
print_status "Total modules processed: ${#json_files[@]}"
