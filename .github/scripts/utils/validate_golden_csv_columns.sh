#!/usr/bin/env bash
set -euo pipefail

# validate_golden_csv_columns.sh
# Validates that golden bandwidth CSV files have the expected column headers
# Automatically discovers all *bandwidth*.csv files in the golden directory

GOLDEN_DIR="tests/tt_metal/tt_metal/perf_microbenchmark/routing/golden"

# Expected header for all bandwidth CSV files (exact match required)
EXPECTED_HEADER="test_name,ftype,ntype,topology,num_devices,num_links,packet_size,iterations,avg_cycles,avg_packets_per_s,avg_bandwidth_gigabytes_per_s,bw_min_gigabytes_per_s,bw_max_gigabytes_per_s,bw_std_dev_gigabytes_per_s,tolerance_percent,max_packet_size"

# Function to check if header matches expected format
validate_header() {
    local header="$1"
    local file="$2"

    # Check for empty header
    if [[ -z "$header" ]]; then
        echo "❌ FAIL: $file - Header is empty"
        return 1
    fi

    # Exact string match
    if [[ "$header" == "$EXPECTED_HEADER" ]]; then
        echo "✅ PASS: $file"
        return 0
    else
        echo "❌ FAIL: $file"
        echo "   Expected: $EXPECTED_HEADER"
        echo "   Actual:   $header"
        return 1
    fi
}

EXIT_CODE=0

# Auto-discover all bandwidth CSV files
echo "Searching for bandwidth CSV files in: $GOLDEN_DIR"
shopt -s nullglob  # Handle case where no files match
CSV_FILES=("$GOLDEN_DIR"/*bandwidth*.csv)
shopt -u nullglob

# Check if any files were found
if [[ ${#CSV_FILES[@]} -eq 0 ]]; then
    echo "❌ ERROR: No bandwidth CSV files found in $GOLDEN_DIR"
    echo "   If the directory was moved or renamed, please update the script path."
    exit 1
fi

echo "Found ${#CSV_FILES[@]} bandwidth CSV file(s)"
echo ""

# Validate each discovered file
for csv_file in "${CSV_FILES[@]}"; do
    if [[ ! -f "$csv_file" ]]; then
        echo "❌ ERROR: File not found: $csv_file"
        EXIT_CODE=1
        continue
    fi

    # Read the first line of the CSV
    header=$(head -n 1 "$csv_file")

    # Validate the header
    if ! validate_header "$header" "$csv_file"; then
        EXIT_CODE=1
    fi
done

echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "✅ All golden bandwidth CSV files have correct column headers"
else
    echo "❌ Some golden bandwidth CSV files have incorrect or missing column headers"
fi

exit $EXIT_CODE
