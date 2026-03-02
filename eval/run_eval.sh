#!/bin/bash
# run_eval.sh - Repeatable agentic workflow evaluation
#
# Drop this script anywhere and run it. It clones tt-metal, builds, and
# runs claude against a prompt file, then optionally runs a test.
#
# Usage:
#   ./run_eval.sh <prompt_file> --branch BRANCH [--runs N] [--test TEST_PATH] [--base-dir DIR]
#
# Example:
#   ./run_eval.sh prompts/create_reduce_avg.txt \
#       --branch mstaletovic/NoPlanner \
#       --runs 3 \
#       --test tests/ttnn/unit_tests/operations/reduce_avg_w_rm/test_reduce_avg_w_rm.py
#
# Each run gets its own clone at:
#   <base-dir>/eval_<branch>_<timestamp>_<run>_<random>/tt-metal/
#
# Results are collected into:
#   <base-dir>/eval_results/<timestamp>/

set -euo pipefail

REPO_URL="https://github.com/tenstorrent/tt-metal.git"
DEFAULT_RUNS=1
DEFAULT_BASE_DIR="/localdev/mstaletovic"

# --- Parse arguments ---
PROMPT_FILE=""
NUM_RUNS="$DEFAULT_RUNS"
BRANCH=""
TEST_PATH=""
BASE_DIR="$DEFAULT_BASE_DIR"

show_usage() {
    echo "Usage: $0 <prompt_file> --branch BRANCH [--runs N] [--test TEST_PATH] [--base-dir DIR]" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs)      NUM_RUNS="$2"; shift 2 ;;
        --branch)    BRANCH="$2"; shift 2 ;;
        --test)      TEST_PATH="$2"; shift 2 ;;
        --base-dir)  BASE_DIR="$2"; shift 2 ;;
        -h|--help)   show_usage ;;
        -*)          echo "Unknown option: $1" >&2; show_usage ;;
        *)
            if [[ -z "$PROMPT_FILE" ]]; then
                PROMPT_FILE="$1"; shift
            else
                echo "Unexpected argument: $1" >&2; show_usage
            fi
            ;;
    esac
done

if [[ -z "$PROMPT_FILE" ]]; then
    echo "Error: prompt file is required" >&2; show_usage
fi
if [[ -z "$BRANCH" ]]; then
    echo "Error: --branch is required" >&2; show_usage
fi

# Resolve prompt file to absolute path before we cd anywhere
PROMPT_FILE="$(cd "$(dirname "$PROMPT_FILE")" && pwd)/$(basename "$PROMPT_FILE")"
if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "Error: Prompt file not found: $PROMPT_FILE" >&2
    exit 1
fi

PROMPT_CONTENT="$(cat "$PROMPT_FILE")"

# --- Setup results directory ---
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
# Sanitize branch name for use in directory names (replace / with _)
BRANCH_SLUG="$(echo "$BRANCH" | tr '/' '_')"
RESULTS_DIR="${BASE_DIR}/eval_results/${BRANCH_SLUG}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "=== Eval Configuration ==="
echo "Prompt:    $PROMPT_FILE"
echo "Branch:    $BRANCH"
echo "Runs:      $NUM_RUNS"
echo "Test:      ${TEST_PATH:-<none - will check claude exit code>}"
echo "Base dir:  $BASE_DIR"
echo "Results:   $RESULTS_DIR"
echo "=========================="
echo ""

# --- Summary tracking ---
PASS_COUNT=0
FAIL_COUNT=0
ERROR_COUNT=0

run_single() {
    local run_id="$1"
    local run_tag="${BRANCH_SLUG}_${TIMESTAMP}_run${run_id}_$$_${RANDOM}"
    local clone_dir="${BASE_DIR}/eval_${run_tag}/tt-metal"
    local log_dir="${RESULTS_DIR}/run_${run_id}"
    mkdir -p "$log_dir"

    echo "--- Run ${run_id}/${NUM_RUNS} ---"
    echo "[${run_id}] Clone dir: $clone_dir"

    # 1. Clone
    echo "[${run_id}] Cloning ${BRANCH}..."
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$clone_dir" \
        > "${log_dir}/clone.log" 2>&1

    # 2. Init submodules
    echo "[${run_id}] Initializing submodules..."
    git -C "$clone_dir" submodule update --init --recursive \
        > "${log_dir}/submodules.log" 2>&1

    # 3. Build
    echo "[${run_id}] Building metal (this takes a while)..."
    (cd "$clone_dir" && ./build_metal.sh --enable-ccache) \
        > "${log_dir}/build.log" 2>&1

    echo "[${run_id}] Creating python env..."
    (cd "$clone_dir" && ./create_venv.sh --force) \
        > "${log_dir}/venv.log" 2>&1

    # 4. Run claude
    echo "[${run_id}] Running claude..."
    local claude_exit=0
    (cd "$clone_dir" && env -u CLAUDECODE claude -p \
        --dangerously-skip-permissions \
        --output-format json \
        --max-turns 100 \
        --model opus \
        "$PROMPT_CONTENT" \
    ) > "${log_dir}/claude_output.json" 2> "${log_dir}/claude_stderr.log" || claude_exit=$?

    echo "[${run_id}] Claude exited with code: $claude_exit"

    # 5. Run test (if specified)
    local test_exit=0
    if [[ -n "$TEST_PATH" ]]; then
        echo "[${run_id}] Running test: $TEST_PATH"
        (cd "$clone_dir" && source python_env/bin/activate && \
            ./tt-test.sh "$TEST_PATH" \
        ) > "${log_dir}/test_output.log" 2>&1 || test_exit=$?

        echo "[${run_id}] Test exited with code: $test_exit"
    else
        test_exit=$claude_exit
    fi

    # 6. Record result
    if [[ $test_exit -eq 0 ]]; then
        echo "PASS" > "${log_dir}/result.txt"
        echo "[${run_id}] PASS"
        ((PASS_COUNT++))
    elif [[ $test_exit -eq 2 ]]; then
        echo "HANG" > "${log_dir}/result.txt"
        echo "[${run_id}] HANG"
        ((ERROR_COUNT++))
    else
        echo "FAIL" > "${log_dir}/result.txt"
        echo "[${run_id}] FAIL"
        ((FAIL_COUNT++))
    fi

    echo ""
}

# --- Run evaluations ---
for i in $(seq 1 "$NUM_RUNS"); do
    run_single "$i"
done

# --- Summary ---
TOTAL=$((PASS_COUNT + FAIL_COUNT + ERROR_COUNT))

cat > "${RESULTS_DIR}/summary.txt" <<EOF
=== Evaluation Summary ===
Timestamp: $TIMESTAMP
Prompt:    $PROMPT_FILE
Branch:    $BRANCH
Test:      ${TEST_PATH:-<claude exit code>}

Results:   $PASS_COUNT/$TOTAL passed
  PASS:    $PASS_COUNT
  FAIL:    $FAIL_COUNT
  ERROR:   $ERROR_COUNT
===========================
EOF

cat "${RESULTS_DIR}/summary.txt"

# Exit non-zero if any run failed
[[ $FAIL_COUNT -eq 0 && $ERROR_COUNT -eq 0 ]]
