#!/bin/bash
# run_eval.sh - Repeatable agentic workflow evaluation
#
# Drop this script anywhere and run it. It clones tt-metal, builds, and
# runs claude against prompt file(s). All runs execute in parallel, and
# claude starts immediately while the build runs in the background.
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
TIMESTAMP_SHORT="$(date +%m%d_%H%M)"
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

run_single() {
    local prompt_file="$1"
    local run_id="$2"
    local prompt_name
    prompt_name="$(basename "$prompt_file" .txt)"
    local short_hash
    short_hash="$(printf '%04x' $(( ($$*31 + RANDOM) % 65536 )) )"
    local run_tag="${prompt_name}_r${run_id}_${TIMESTAMP_SHORT}_${short_hash}"
    local clone_dir="${BASE_DIR}/eval_${run_tag}/tt-metal"
    local log_dir="${RESULTS_DIR}/${prompt_name}/run_${run_id}"
    mkdir -p "$log_dir"

    local prompt_content
    prompt_content="$(cat "$prompt_file")"

    local start_time=$SECONDS

    echo "[${prompt_name}:${run_id}] Starting..."

    # 1. Clone and create a unique branch from the source branch
    local run_branch="eval/${prompt_name}_r${run_id}_${TIMESTAMP_SHORT}"
    echo "[${prompt_name}:${run_id}] Cloning ${BRANCH} -> ${run_branch}"
    git clone --branch "$BRANCH" "$REPO_URL" "$clone_dir" \
        > "${log_dir}/clone.log" 2>&1
    git -C "$clone_dir" checkout -b "$run_branch" \
        >> "${log_dir}/clone.log" 2>&1

    # 2. Init submodules
    echo "[${prompt_name}:${run_id}] Initializing submodules..."
    git -C "$clone_dir" submodule update --init --recursive \
        > "${log_dir}/submodules.log" 2>&1

    # 3. Everything from here runs in a subshell with isolated env vars.
    local claude_exit=0
    (
        cd "$clone_dir"
        export TT_METAL_HOME="$clone_dir"
        export PYTHONPATH="$clone_dir"
        export TT_METAL_ENV=dev
        export TT_METAL_CACHE="$clone_dir/built"
        export TT_METAL_CCACHE_KERNEL_SUPPORT=1
        unset CLAUDECODE

        # Sentinel files for build stages
        local build_sentinel="${clone_dir}/.build_complete"
        local venv_sentinel="${clone_dir}/.venv_complete"
        local infra_failed="${clone_dir}/.infra_failed"

        # 4. Build and venv in parallel
        (
            echo "[${prompt_name}:${run_id}] Building metal..."
            if ! ./build_metal.sh --enable-ccache > "${log_dir}/build.log" 2>&1; then
                echo "BUILD_FAIL" > "${log_dir}/result.txt"
                touch "$infra_failed"
                exit 1
            fi
            touch "$build_sentinel"
            echo "[${prompt_name}:${run_id}] Build complete"
        ) &
        local build_pid=$!

        (
            echo "[${prompt_name}:${run_id}] Creating python env..."
            if ! ./create_venv.sh --force > "${log_dir}/venv.log" 2>&1; then
                echo "VENV_FAIL" > "${log_dir}/result.txt"
                touch "$infra_failed"
                exit 1
            fi
            touch "$venv_sentinel"
            echo "[${prompt_name}:${run_id}] Venv complete"
        ) &
        local venv_pid=$!

        # 5. Install a gate script that blocks until build+venv are done.
        #    tt-test.sh sources python_env/bin/activate and needs built artifacts,
        #    so any test execution must wait. Claude's early turns (reading code,
        #    planning, writing) don't need the build and run freely.
        mkdir -p "${clone_dir}/.eval"
        cat > "${clone_dir}/.eval/wait_for_build.sh" <<GATE
#!/bin/bash
# Blocks until both build and venv are complete.
# If either background job failed, exits cleanly so Claude
# can rebuild on its own and tests will work.
while [ ! -f "${build_sentinel}" ]; do
    if [ -f "${infra_failed}" ]; then
        echo "NOTE: Background build failed. Proceeding (agent may rebuild)." >&2
        break
    fi
    sleep 5
done
while [ ! -f "${venv_sentinel}" ]; do
    if [ -f "${infra_failed}" ]; then
        echo "NOTE: Background venv failed. Proceeding (agent may recreate)." >&2
        break
    fi
    sleep 5
done
GATE
        chmod +x "${clone_dir}/.eval/wait_for_build.sh"

        # Wrap tt-test.sh: prepend a build gate so tests block until ready
        if [[ -f "${clone_dir}/tt-test.sh" ]]; then
            local original_test_sh
            original_test_sh="$(cat "${clone_dir}/tt-test.sh")"
            cat > "${clone_dir}/tt-test.sh" <<WRAPPER
#!/bin/bash
# Auto-generated wrapper: wait for build before running tests
source "${clone_dir}/.eval/wait_for_build.sh" || exit \$?
# --- Original tt-test.sh below ---
$(echo "$original_test_sh" | tail -n +2)
WRAPPER
            chmod +x "${clone_dir}/tt-test.sh"
        fi

        # 6. Run claude immediately (doesn't need build for early turns)
        echo "[${prompt_name}:${run_id}] Running claude (build in background)..."
        claude -p \
            --dangerously-skip-permissions \
            --output-format json \
            --max-turns 150 \
            --model opus \
            "$prompt_content" \
            > "${log_dir}/claude_output.json" 2> "${log_dir}/claude_stderr.log"

        # Clean up: wait for background processes to finish (may already be done)
        wait "$build_pid" 2>/dev/null || true
        wait "$venv_pid" 2>/dev/null || true
    ) || claude_exit=$?

    echo "[${prompt_name}:${run_id}] Exited with code: $claude_exit"

    # 7. Record result
    local elapsed=$(( SECONDS - start_time ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))

    local result
    if [[ $claude_exit -eq 0 ]]; then
        result="PASS"
    elif [[ -f "${log_dir}/result.txt" ]] && grep -q "_FAIL" "${log_dir}/result.txt"; then
        result="$(cat "${log_dir}/result.txt")"
    else
        result="FAIL (exit $claude_exit)"
    fi
    echo "$result" > "${log_dir}/result.txt"
    echo "[${prompt_name}:${run_id}] ${result} (${mins}m ${secs}s)"
    echo "${elapsed}" > "${log_dir}/duration_seconds.txt"
}

# --- Launch all runs in parallel ---
PIDS=()
for prompt_file in "${PROMPT_FILES[@]}"; do
    for run in $(seq 1 "$NUM_RUNS"); do
        run_single "$prompt_file" "$run" &
        PIDS+=($!)
    done
done

echo "Launched ${#PIDS[@]} parallel runs. Waiting for all to complete..."
echo ""

# Wait for all background jobs
for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

# --- Collect results from files ---
PASS_COUNT=0
FAIL_COUNT=0
ERROR_COUNT=0

while IFS= read -r result_file; do
    result="$(cat "$result_file")"
    case "$result" in
        PASS)          PASS_COUNT=$((PASS_COUNT + 1)) ;;
        *_FAIL)        ERROR_COUNT=$((ERROR_COUNT + 1)) ;;
        *)             FAIL_COUNT=$((FAIL_COUNT + 1)) ;;
    esac
done < <(find "$RESULTS_DIR" -name "result.txt" -type f)

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

echo ""
cat "${RESULTS_DIR}/summary.txt"

# Exit non-zero if any run failed
[[ $FAIL_COUNT -eq 0 ]]
