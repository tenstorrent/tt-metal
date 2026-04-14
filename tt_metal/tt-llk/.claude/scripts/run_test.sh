#!/bin/bash
# LLK Test Runner Script
# Usage: ENV_SETUP=0 COMPILED=1 RUN_TEST=1 FILE_NAME="test_name.py" .claude/scripts/run_test.sh
# Must be run from the tests/ directory.

set -euo pipefail

# Defaults
ENV_SETUP="${ENV_SETUP:-0}"
COMPILED="${COMPILED:-1}"
RUN_TEST="${RUN_TEST:-1}"
FILE_NAME="${FILE_NAME:-}"
TEST_PATH="${TEST_PATH:-}"
QUIET="${QUIET:-1}"
COVERAGE="${COVERAGE:-0}"
PARALLEL_JOBS="${PARALLEL_JOBS:-10}"
FAIL_FAST="${FAIL_FAST:-1}"
PYTEST_ARGS="${PYTEST_ARGS:-}"

LOG_DIR="/tmp/llk_test"
mkdir -p "$LOG_DIR"

# Determine test target
if [[ -n "$TEST_PATH" ]]; then
    TEST_TARGET="$TEST_PATH"
elif [[ -n "$FILE_NAME" ]]; then
    TEST_TARGET="python_tests/$FILE_NAME"
else
    TEST_TARGET="python_tests/"
fi

# Build pytest flags
PYTEST_FLAGS=""
if [[ "$FAIL_FAST" == "1" ]]; then
    PYTEST_FLAGS="$PYTEST_FLAGS -x"
fi
if [[ "$COVERAGE" == "1" ]]; then
    PYTEST_FLAGS="$PYTEST_FLAGS --coverage"
fi
if [[ -n "$PYTEST_ARGS" ]]; then
    PYTEST_FLAGS="$PYTEST_FLAGS $PYTEST_ARGS"
fi

# Activate venv if it exists
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
fi

# Step 1: Environment setup
if [[ "$ENV_SETUP" == "1" ]]; then
    echo "=== Setting up environment ==="
    if [[ -f "./setup_testing_env.sh" ]]; then
        ./setup_testing_env.sh
    else
        echo "WARNING: setup_testing_env.sh not found"
    fi
fi

# Step 2: Compile (producer)
if [[ "$COMPILED" == "1" ]]; then
    echo "=== Compiling: $TEST_TARGET ==="
    COMPILE_CMD="pytest --compile-producer -n $PARALLEL_JOBS $PYTEST_FLAGS $TEST_TARGET"

    if [[ "$QUIET" == "1" ]]; then
        eval "$COMPILE_CMD" > "$LOG_DIR/compile.log" 2>&1 || {
            echo "COMPILE FAILED — see $LOG_DIR/compile.log"
            tail -20 "$LOG_DIR/compile.log"
            exit 1
        }
        echo "Compile: OK (log: $LOG_DIR/compile.log)"
    else
        eval "$COMPILE_CMD" 2>&1 | tee "$LOG_DIR/compile.log" || {
            echo "COMPILE FAILED — see $LOG_DIR/compile.log"
            exit 1
        }
    fi
fi

# Step 3: Run tests (consumer)
if [[ "$RUN_TEST" == "1" ]]; then
    echo "=== Running: $TEST_TARGET ==="
    RUN_CMD="pytest --compile-consumer $PYTEST_FLAGS $TEST_TARGET"

    if [[ "$QUIET" == "1" ]]; then
        eval "$RUN_CMD" > "$LOG_DIR/run.log" 2>&1 || {
            echo "TESTS FAILED — see $LOG_DIR/run.log"
            tail -10 "$LOG_DIR/run.log"
            exit 1
        }
        echo "Run: OK (log: $LOG_DIR/run.log)"
        tail -10 "$LOG_DIR/run.log"
    else
        eval "$RUN_CMD" 2>&1 | tee "$LOG_DIR/run.log" || {
            echo "TESTS FAILED — see $LOG_DIR/run.log"
            exit 1
        }
    fi
fi

echo "=== Done ==="
