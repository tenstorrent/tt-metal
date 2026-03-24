#!/bin/bash
# OLMo ISL Sweep Script
# Runs ISL 128, 1K, 2K, 4K tests and captures generated output for coherency verification

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Output directory
OUTPUT_DIR="${TT_METAL_ROOT}/olmo_isl_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "OLMo ISL Sweep Test"
echo "Output directory: $OUTPUT_DIR"
echo "=============================================="

# Setup environment
cd "$TT_METAL_ROOT"
export TT_METAL_HOME="$TT_METAL_ROOT"
export PYTHONPATH="$TT_METAL_ROOT"
source python_env/bin/activate

# Use default model path or environment variable
export HF_MODEL="${HF_MODEL:-~/models/models--allenai--Olmo-3.1-32B-Think}"
export LINE_RS=1

echo "Using model: $HF_MODEL"
echo ""

# ISL configurations to test
ISLS=("128" "1k" "2k" "4k")

# Summary file
SUMMARY_FILE="$OUTPUT_DIR/summary.txt"
echo "OLMo ISL Sweep Summary - $(date)" > "$SUMMARY_FILE"
echo "==========================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Run each ISL test
for isl in "${ISLS[@]}"; do
    echo "----------------------------------------------"
    echo "Running ISL ${isl}..."
    echo "----------------------------------------------"

    LOG_FILE="$OUTPUT_DIR/isl_${isl}.log"
    COHERENCY_FILE="$OUTPUT_DIR/isl_${isl}_output.txt"

    # Run test and capture output
    if timeout 300 pytest models/demos/llama3_70b_galaxy/demo/demo_olmo_decode.py::test_olmo_demo \
        -v -k "isl-${isl}-b1" -s 2>&1 | tee "$LOG_FILE"; then
        STATUS="PASS"
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 124 ]; then
            STATUS="TIMEOUT"
        else
            STATUS="FAIL"
        fi
    fi

    # Extract coherency section
    echo "=== ISL ${isl} Generated Output ===" > "$COHERENCY_FILE"
    sed -n '/--- User 0 ---/,/PERFORMANCE SUMMARY/p' "$LOG_FILE" >> "$COHERENCY_FILE" 2>/dev/null || echo "No output captured" >> "$COHERENCY_FILE"

    # Extract performance metrics
    TTFT=$(grep "Time to First Token" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP '[\d.]+(?= ms)' || echo "N/A")
    TOKS=$(grep "tok/s/user" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP '[\d.]+(?= tok/s/user)' || echo "N/A")

    # Add to summary
    echo "ISL ${isl}: ${STATUS}" >> "$SUMMARY_FILE"
    echo "  TTFT: ${TTFT} ms" >> "$SUMMARY_FILE"
    echo "  Throughput: ${TOKS} tok/s/user" >> "$SUMMARY_FILE"
    echo "  Log: isl_${isl}.log" >> "$SUMMARY_FILE"
    echo "  Output: isl_${isl}_output.txt" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"

    echo "ISL ${isl}: ${STATUS} (TTFT: ${TTFT}ms, ${TOKS} tok/s)"
    echo ""
done

echo "----------------------------------------------"
echo "Extracting all generated outputs..."
echo "----------------------------------------------"

# Create combined output file
COMBINED_FILE="$OUTPUT_DIR/all_outputs.txt"
echo "OLMo ISL Sweep - All Generated Outputs" > "$COMBINED_FILE"
echo "=======================================" >> "$COMBINED_FILE"
echo "Generated: $(date)" >> "$COMBINED_FILE"
echo "" >> "$COMBINED_FILE"

for isl in "${ISLS[@]}"; do
    echo "" >> "$COMBINED_FILE"
    echo "=============================================" >> "$COMBINED_FILE"
    echo "ISL ${isl}" >> "$COMBINED_FILE"
    echo "=============================================" >> "$COMBINED_FILE"
    cat "$OUTPUT_DIR/isl_${isl}_output.txt" >> "$COMBINED_FILE" 2>/dev/null || echo "No output" >> "$COMBINED_FILE"
done

echo ""
echo "=============================================="
echo "ISL Sweep Complete!"
echo "=============================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Files:"
echo "  - summary.txt        : Performance summary"
echo "  - all_outputs.txt    : All generated outputs"
echo "  - isl_*.log          : Full test logs"
echo "  - isl_*_output.txt   : Individual outputs"
echo ""
echo "Summary:"
cat "$SUMMARY_FILE"
