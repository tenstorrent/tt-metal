#!/bin/bash
# Stress test for tt-test.sh infrastructure
#
# Launches a mix of passing, failing (PCC), hanging, and mixed tests concurrently.
# All go through tt-test.sh's flock serialization.
#
# Note: tt-test.sh uses -x (stop on first failure). For mixed tests, this means
# passing tests before the first failure will run, then it stops.
#
# Usage: bash tests/ttnn/unit_tests/test_stress_infra/run_stress.sh

SCRIPT="./tt-test.sh"
TEST_DIR="tests/ttnn/unit_tests/test_stress_infra"
LOG_DIR="/tmp/tt-stress"
rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"

echo "=== TT-TEST STRESS TEST ==="
echo ""

N=0

launch() {
    local name=$1; shift
    N=$((N + 1))
    "$@" > "$LOG_DIR/${name}.log" 2>&1 &
    PIDS[$N]=$!
    NAMES[$N]="$name"
    EXPECT[$N]="$EXPECTED"
    echo "  [$name] PID=${PIDS[$N]}"
}

# --- Pure pass (2x) ---
EXPECTED="0"
launch pass_1      $SCRIPT "$TEST_DIR/test_pass.py"
launch pass_2      $SCRIPT "$TEST_DIR/test_pass.py"

# --- Pure PCC fail (2x) ---
EXPECTED="1"
launch pcc_fail_1  $SCRIPT "$TEST_DIR/test_pcc_fail.py"
launch pcc_fail_2  $SCRIPT "$TEST_DIR/test_pcc_fail.py"

# --- Pure hang (2x) ---
EXPECTED="2"
launch hang_1      $SCRIPT "$TEST_DIR/test_hang.py"
launch hang_2      $SCRIPT "$TEST_DIR/test_hang.py"

# --- Mixed pass+fail (2x) — stops at first failure due to -x ---
EXPECTED="1"
launch mixed_1     $SCRIPT "$TEST_DIR/test_mixed.py"
launch mixed_2     $SCRIPT "$TEST_DIR/test_mixed.py"

# --- Mixed: only the passing tests from the mixed file (2x) ---
EXPECTED="0"
launch mixed_pass_only_1  $SCRIPT "$TEST_DIR/test_mixed.py" -k "test_add_pass or test_multiply_pass"
launch mixed_pass_only_2  $SCRIPT "$TEST_DIR/test_mixed.py" -k "test_add_pass or test_multiply_pass"

# --- NoC sanitizer violation (1x) ---
# Passes silently in basic mode (misaligned read goes undetected).
# Only caught in --dev mode where watcher NoC sanitizer is active.
# In basic mode this is expected to PASS (the bug is invisible).
EXPECTED="0"
launch noc_sanitizer_1  $SCRIPT "$TEST_DIR/test_noc_sanitizer.py"

echo ""
echo "Launched $N instances. Waiting for all to complete..."
echo ""

# Wait and collect
for i in $(seq 1 $N); do
    wait ${PIDS[$i]}
    EXITS[$i]=$?
done

# Results
echo "=== RESULTS ==="
printf "%-20s %-6s %-40s %s\n" "Test" "Exit" "Pytest counts" "Match"
printf "%-20s %-6s %-40s %s\n" "----" "----" "-------------" "-----"
for i in $(seq 1 $N); do
    EXIT=${EXITS[$i]}
    EXPECTED_EXIT=${EXPECT[$i]}

    # Get pass/fail/error counts
    COUNTS=$(grep -oE "[0-9]+ passed|[0-9]+ failed|[0-9]+ deselected|[0-9]+ error" "$LOG_DIR/${NAMES[$i]}.log" 2>/dev/null | tr '\n' ', ' | sed 's/, $//')

    if [[ "$EXIT" == "$EXPECTED_EXIT" ]]; then
        MATCH="ok"
    else
        MATCH="MISMATCH (expected $EXPECTED_EXIT)"
    fi

    printf "%-20s %-6s %-40s %s\n" "${NAMES[$i]}" "$EXIT" "$COUNTS" "$MATCH"
done

# Show execution order via lock acquisition timestamps
echo ""
echo "=== EXECUTION ORDER (flock serialization) ==="
for i in $(seq 1 $N); do
    ACQUIRED=$(grep "Device lock acquired" "$LOG_DIR/${NAMES[$i]}.log" 2>/dev/null | head -1)
    RESULT=$(grep "TT_TEST_RESULT" "$LOG_DIR/${NAMES[$i]}.log" 2>/dev/null | tail -1)
    # Extract just the timestamp from the log line if present
    ACQ_TIME=$(echo "$ACQUIRED" | grep -oE "[0-9]{2}:[0-9]{2}:[0-9]{2}" | head -1)
    RES_TIME=""
    # Try to get the timestamp from nearby log lines
    RESET_LINE=$(grep "Device reset complete\|TT_TEST_RESULT" "$LOG_DIR/${NAMES[$i]}.log" 2>/dev/null | tail -1)
    printf "  %-20s %s\n" "${NAMES[$i]}" "$RESULT"
done

echo ""
echo "Logs: $LOG_DIR/"
