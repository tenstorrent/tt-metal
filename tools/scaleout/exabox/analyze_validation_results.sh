#!/bin/bash

# Script to analyze validation output logs for health status and errors
# Usage: ./analyze_validation_results.sh [directory]
# Default directory: validation_output/

# Set default directory or use provided argument
VALIDATION_DIR="${1:-validation_output}"

# Check if directory exists
if [ ! -d "$VALIDATION_DIR" ]; then
    echo "Error: Directory $VALIDATION_DIR does not exist"
    exit 1
fi

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Validation Results Analysis"
echo "Directory: $VALIDATION_DIR"
echo "Timestamp: $(date)"
echo "=========================================="
echo ""

# Count total log files
TOTAL_FILES=$(find "$VALIDATION_DIR" -name "*.log" -type f | wc -l)
echo "Total log files found: $TOTAL_FILES"
echo ""

# 1. Check for healthy links
echo -e "${GREEN}✓ Checking for 'All Detected Links are healthy'${NC}"
HEALTHY_FILES=$(grep -l "All Detected Links are healthy" "$VALIDATION_DIR"/*.log 2>/dev/null)
HEALTHY_COUNT=$(echo "$HEALTHY_FILES" | grep -c "." 2>/dev/null || echo "0")
echo "  Count: $HEALTHY_COUNT"
if [ $HEALTHY_COUNT -gt 0 ]; then
    echo "  Files:"
    echo "$HEALTHY_FILES" | sed 's/^/    /'
fi
echo ""

# 2. Check for unhealthy links
echo -e "${RED}✗ Checking for 'Found Unhealthy Links'${NC}"
UNHEALTHY_FILES=$(grep -l "Found Unhealthy Links" "$VALIDATION_DIR"/*.log 2>/dev/null)
UNHEALTHY_COUNT=$(echo "$UNHEALTHY_FILES" | grep -c "." 2>/dev/null || echo "0")
echo "  Count: $UNHEALTHY_COUNT"
if [ $UNHEALTHY_COUNT -gt 0 ]; then
    echo "  Files:"
    echo "$UNHEALTHY_FILES" | sed 's/^/    /'
fi
echo ""

# 3. Check for timeout issues
echo -e "${YELLOW}⏱ Checking for 'Timeout (10000 ms) waiting for physical cores to finish'${NC}"
TIMEOUT_FILES=$(grep -l "Timeout (10000 ms) waiting for physical cores to finish" "$VALIDATION_DIR"/*.log 2>/dev/null)
TIMEOUT_COUNT=$(echo "$TIMEOUT_FILES" | grep -c "." 2>/dev/null || echo "0")
echo "  Count: $TIMEOUT_COUNT"
if [ $TIMEOUT_COUNT -gt 0 ]; then
    echo "  Files:"
    echo "$TIMEOUT_FILES" | sed 's/^/    /'
fi
echo ""

# 4. Check for missing port/cable connections
echo -e "${BLUE}🔌 Checking for 'missing port/cable connections'${NC}"
MISSING_CONN_FILES=$(grep -l "missing port/cable connections" "$VALIDATION_DIR"/*.log 2>/dev/null)
MISSING_CONN_COUNT=$(echo "$MISSING_CONN_FILES" | grep -c "." 2>/dev/null || echo "0")
echo "  Count: $MISSING_CONN_COUNT"
if [ $MISSING_CONN_COUNT -gt 0 ]; then
    echo "  Files:"
    echo "$MISSING_CONN_FILES" | sed 's/^/    /'
fi
echo ""

# 5. Check for DRAM training failures
echo -e "${RED}⚠ Checking for 'DRAM training failed'${NC}"
DRAM_FAIL_FILES=$(grep -l "DRAM training failed" "$VALIDATION_DIR"/*.log 2>/dev/null)
DRAM_FAIL_COUNT=$(echo "$DRAM_FAIL_FILES" | grep -c "." 2>/dev/null || echo "0")
echo "  Count: $DRAM_FAIL_COUNT"
if [ $DRAM_FAIL_COUNT -gt 0 ]; then
    echo "  Files:"
    echo "$DRAM_FAIL_FILES" | sed 's/^/    /'
fi
echo ""

# 6. Check for extra port/cable connections
echo -e "${YELLOW}🔗 Checking for 'extra port/cable connections'${NC}"
EXTRA_CONN_FILES=$(grep -l "extra port/cable connections" "$VALIDATION_DIR"/*.log 2>/dev/null)
EXTRA_CONN_COUNT=$(echo "$EXTRA_CONN_FILES" | grep -c "." 2>/dev/null || echo "0")
echo "  Count: $EXTRA_CONN_COUNT"
if [ $EXTRA_CONN_COUNT -gt 0 ]; then
    echo "  Files:"
    echo "$EXTRA_CONN_FILES" | sed 's/^/    /'
fi
echo ""

# 7. Files with none of these strings (indeterminate/incomplete)
echo -e "📋 Checking for files with NONE of these strings (indeterminate/incomplete):"
INDETERMINATE_FILES=""
for file in "$VALIDATION_DIR"/*.log; do
    if [ -f "$file" ]; then
        if ! grep -q "All Detected Links are healthy\|Found Unhealthy Links\|Timeout (10000 ms) waiting for physical cores to finish\|missing port/cable connections\|DRAM training failed\|extra port/cable connections" "$file" 2>/dev/null; then
            INDETERMINATE_FILES="$INDETERMINATE_FILES$file\n"
        fi
    fi
done
INDETERMINATE_COUNT=$(echo -e "$INDETERMINATE_FILES" | grep -c "." 2>/dev/null || echo "0")
echo "  Count: $INDETERMINATE_COUNT"
if [ $INDETERMINATE_COUNT -gt 0 ]; then
    echo "  Files:"
    echo -e "$INDETERMINATE_FILES" | sed 's/^/    /'
fi
echo ""

# Summary
echo "=========================================="
echo "Summary"
echo "=========================================="
echo -e "${GREEN}Healthy links:${NC}           $HEALTHY_COUNT / $TOTAL_FILES"
echo -e "${RED}Unhealthy links:${NC}         $UNHEALTHY_COUNT / $TOTAL_FILES"
echo -e "${YELLOW}Timeout issues:${NC}          $TIMEOUT_COUNT / $TOTAL_FILES"
echo -e "${BLUE}Missing connections:${NC}     $MISSING_CONN_COUNT / $TOTAL_FILES"
echo -e "${RED}DRAM training failures:${NC}  $DRAM_FAIL_COUNT / $TOTAL_FILES"
echo -e "${YELLOW}Extra connections:${NC}       $EXTRA_CONN_COUNT / $TOTAL_FILES"
echo "Indeterminate/Incomplete: $INDETERMINATE_COUNT / $TOTAL_FILES"
echo ""

# Success rate calculation
if [ $TOTAL_FILES -gt 0 ]; then
    SUCCESS_RATE=$(echo "scale=2; $HEALTHY_COUNT * 100 / $TOTAL_FILES" | bc)
    echo "Success Rate: $SUCCESS_RATE%"
fi
