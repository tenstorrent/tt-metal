#!/bin/bash
# run_eval.sh - Repeatable agentic workflow evaluation
#
# Drop this script anywhere and run it. It clones tt-metal, builds, and
# runs claude against prompt file(s).
#
# Usage:
#   ./run_eval.sh <prompt_file_or_dir> [--runs N] [--base-dir DIR]
#
# Run from the tt-metal repo root. The current branch is used as the starting point.
#
# Examples:
#   # Single prompt (on whatever branch you're on)
#   ./eval/run_eval.sh eval/prompts/create_reduce_avg.txt
#
#   # All prompts in a folder (one clone+run per .txt file)
#   ./eval/run_eval.sh eval/prompts/ --runs 3
#
# Each run gets its own clone at:
#   <base-dir>/eval_<branch>_<timestamp>_<prompt>_<run>/tt-metal/
#
# Results are collected into:
#   <base-dir>/eval_results/<branch>_<timestamp>/

set -euo pipefail

DEFAULT_RUNS=1
DEFAULT_BASE_DIR="/localdev/${USER}"

# --- Parse arguments ---
PROMPT_PATH=""
NUM_RUNS="$DEFAULT_RUNS"
BASE_DIR="$DEFAULT_BASE_DIR"

show_usage() {
    echo "Usage: $0 <prompt_file_or_dir> [--runs N] [--base-dir DIR]" >&2
    echo "       Run from the tt-metal repo root." >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs)      NUM_RUNS="$2"; shift 2 ;;
        --base-dir)  BASE_DIR="$2"; shift 2 ;;
        -h|--help)   show_usage ;;
        -*)          echo "Unknown option: $1" >&2; show_usage ;;
        *)
            if [[ -z "$PROMPT_PATH" ]]; then
                PROMPT_PATH="$1"; shift
            else
                echo "Unexpected argument: $1" >&2; show_usage
            fi
            ;;
    esac
done

if [[ -z "$PROMPT_PATH" ]]; then
    echo "Error: prompt file or directory is required" >&2; show_usage
fi

# Infer branch and repo URL from the current git repo
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
REPO_URL="$(git remote get-url origin)"
if [[ -z "$BRANCH" || "$BRANCH" == "HEAD" ]]; then
    echo "Error: Could not determine current branch. Are you in a git repo?" >&2
    exit 1
fi

# --- Collect prompt files ---
PROMPT_FILES=()

if [[ -d "$PROMPT_PATH" ]]; then
    PROMPT_DIR="$(cd "$PROMPT_PATH" && pwd)"
    while IFS= read -r f; do
        PROMPT_FILES+=("$f")
    done < <(find "$PROMPT_DIR" -maxdepth 1 -name '*.txt' -type f | sort)
    if [[ ${#PROMPT_FILES[@]} -eq 0 ]]; then
        echo "Error: No .txt files found in $PROMPT_DIR" >&2
        exit 1
    fi
elif [[ -f "$PROMPT_PATH" ]]; then
    PROMPT_FILES+=("$(cd "$(dirname "$PROMPT_PATH")" && pwd)/$(basename "$PROMPT_PATH")")
else
    echo "Error: Not a file or directory: $PROMPT_PATH" >&2
    exit 1
fi

# --- Setup results directory ---
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BRANCH_SLUG="$(echo "$BRANCH" | tr '/' '_')"
RESULTS_DIR="${BASE_DIR}/eval_results/${BRANCH_SLUG}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "=== Eval Configuration ==="
echo "Prompts:   ${#PROMPT_FILES[@]} file(s)"
for pf in "${PROMPT_FILES[@]}"; do
    echo "           - $(basename "$pf")"
done
echo "Branch:    $BRANCH"
echo "Runs:      $NUM_RUNS per prompt"
echo "Base dir:  $BASE_DIR"
echo "Results:   $RESULTS_DIR"
echo "=========================="
echo ""

# --- Summary tracking ---
PASS_COUNT=0
FAIL_COUNT=0
ERROR_COUNT=0

run_single() {
    local prompt_file="$1"
    local run_id="$2"
    local prompt_name
    prompt_name="$(basename "$prompt_file" .txt)"
    local run_tag="${BRANCH_SLUG}_${TIMESTAMP}_${prompt_name}_run${run_id}_$$_${RANDOM}"
    local clone_dir="${BASE_DIR}/eval_${run_tag}/tt-metal"
    local log_dir="${RESULTS_DIR}/${prompt_name}/run_${run_id}"
    mkdir -p "$log_dir"

    local prompt_content
    prompt_content="$(cat "$prompt_file")"

    local start_time=$SECONDS

    echo "--- ${prompt_name} run ${run_id}/${NUM_RUNS} ---"
    echo "[${prompt_name}:${run_id}] Clone dir: $clone_dir"

    # 1. Clone and create a unique branch from the source branch
    local run_branch="eval/${prompt_name}_run${run_id}_${TIMESTAMP}"
    echo "[${prompt_name}:${run_id}] Cloning ${BRANCH} -> ${run_branch}"
    git clone --branch "$BRANCH" "$REPO_URL" "$clone_dir" \
        > "${log_dir}/clone.log" 2>&1
    git -C "$clone_dir" checkout -b "$run_branch" \
        >> "${log_dir}/clone.log" 2>&1

    # 2. Init submodules
    echo "[${prompt_name}:${run_id}] Initializing submodules..."
    git -C "$clone_dir" submodule update --init --recursive \
        > "${log_dir}/submodules.log" 2>&1

    # 3. Build and run claude in an isolated env (mirrors .envrc)
    #    Everything runs in a subshell so env vars don't leak between runs.
    local claude_exit=0
    (
        cd "$clone_dir"
        export TT_METAL_HOME="$clone_dir"
        export PYTHONPATH="$clone_dir"
        export TT_METAL_ENV=dev
        export TT_METAL_CACHE="$clone_dir/built"
        export TT_METAL_CCACHE_KERNEL_SUPPORT=1
        unset CLAUDECODE

        echo "[${prompt_name}:${run_id}] Building metal (this takes a while)..."
        if ! ./build_metal.sh --enable-ccache > "${log_dir}/build.log" 2>&1; then
            echo "BUILD_FAIL" > "${log_dir}/result.txt"
            exit 2
        fi

        echo "[${prompt_name}:${run_id}] Creating python env..."
        if ! ./create_venv.sh --force > "${log_dir}/venv.log" 2>&1; then
            echo "VENV_FAIL" > "${log_dir}/result.txt"
            exit 2
        fi

        echo "[${prompt_name}:${run_id}] Running claude..."
        claude -p \
            --dangerously-skip-permissions \
            --output-format json \
            --max-turns 100 \
            --model opus \
            "$prompt_content" \
            > "${log_dir}/claude_output.json" 2> "${log_dir}/claude_stderr.log"
    ) || claude_exit=$?

    echo "[${prompt_name}:${run_id}] Exited with code: $claude_exit"

    # 4. Record result
    local elapsed=$(( SECONDS - start_time ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))

    local result
    if [[ $claude_exit -eq 0 ]]; then
        result="PASS"
        PASS_COUNT=$((PASS_COUNT + 1))
    elif [[ -f "${log_dir}/result.txt" ]] && grep -q "_FAIL" "${log_dir}/result.txt"; then
        result="$(cat "${log_dir}/result.txt")"
        ERROR_COUNT=$((ERROR_COUNT + 1))
    else
        result="FAIL (exit $claude_exit)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo "$result" > "${log_dir}/result.txt"
    echo "[${prompt_name}:${run_id}] ${result} (${mins}m ${secs}s)"
    echo "${elapsed}" > "${log_dir}/duration_seconds.txt"

    echo ""
}

# --- Run evaluations ---
for prompt_file in "${PROMPT_FILES[@]}"; do
    for run in $(seq 1 "$NUM_RUNS"); do
        run_single "$prompt_file" "$run"
    done
done

# --- Summary ---
TOTAL=$((PASS_COUNT + FAIL_COUNT + ERROR_COUNT))

cat > "${RESULTS_DIR}/summary.txt" <<EOF
=== Evaluation Summary ===
Timestamp: $TIMESTAMP
Prompts:   ${#PROMPT_FILES[@]} file(s)
Branch:    $BRANCH
Runs:      $NUM_RUNS per prompt

Results:   $PASS_COUNT/$TOTAL passed
  PASS:    $PASS_COUNT
  FAIL:    $FAIL_COUNT
  INFRA:   $ERROR_COUNT
===========================
EOF

cat "${RESULTS_DIR}/summary.txt"

# Exit non-zero if any run failed
[[ $FAIL_COUNT -eq 0 ]]
