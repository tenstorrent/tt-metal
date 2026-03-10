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
# Directory structure:
#   <base-dir>/<YYYY_MM_DD>/<HHMM>_<branch>/
#     clones/<prompt_name>_run<N>/tt-metal/
#     results/<prompt_name>/run_<N>/
#     results/summary.txt

set -euo pipefail

DEFAULT_RUNS=1
DEFAULT_BASE_DIR="/localdev/${USER}"
MONITOR_INTERVAL=30  # seconds between progress updates

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
        --runs)              NUM_RUNS="$2"; shift 2 ;;
        --base-dir)          BASE_DIR="$2"; shift 2 ;;
        --monitor-interval)  MONITOR_INTERVAL="$2"; shift 2 ;;
        -h|--help)           show_usage ;;
        -*)                  echo "Unknown option: $1" >&2; show_usage ;;
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

# Infer branch, commit, and repo URL from the current git repo
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
STARTING_COMMIT="$(git rev-parse HEAD)"
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

# --- Setup session directory ---
DATE_STAMP="$(date +%Y_%m_%d)"
TIME_STAMP="$(date +%H%M)"
BRANCH_SLUG="$(echo "$BRANCH" | tr '/' '_')"
SESSION_DIR="${BASE_DIR}/${DATE_STAMP}/${TIME_STAMP}_${BRANCH_SLUG}"
RESULTS_DIR="${SESSION_DIR}/results"
mkdir -p "$RESULTS_DIR"

echo "=== Eval Configuration ==="
echo "Prompts:   ${#PROMPT_FILES[@]} file(s)"
for pf in "${PROMPT_FILES[@]}"; do
    echo "           - $(basename "$pf")"
done
echo "Branch:    $BRANCH"
echo "Runs:      $NUM_RUNS per prompt"
echo "Session:   $SESSION_DIR"
echo "Results:   $RESULTS_DIR"
echo "=========================="
echo ""

# --- Progress monitor ---
MONITOR_PID=""

format_elapsed() {
    local secs="$1"
    printf "%dm%02ds" $((secs / 60)) $((secs % 60))
}

# Detect the pipeline phase from the latest git commit message in a clone
detect_phase() {
    local clone_dir="$1"
    if [[ ! -d "$clone_dir/.git" ]]; then
        echo "cloning"
        return
    fi

    local last_msg
    last_msg="$(git -C "$clone_dir" log --oneline -1 2>/dev/null | sed 's/^[a-f0-9]* //')"

    # Check build/venv sentinels
    local build_done=false venv_done=false
    [[ -f "$clone_dir/.build_complete" ]] && build_done=true
    [[ -f "$clone_dir/.venv_complete" ]] && venv_done=true

    case "$last_msg" in
        *"[ttnn-self-reflection]"*)    echo "self-reflection" ;;
        *"[create-op] report:"*)       echo "reporting" ;;
        *"[ttnn-kernel-writer-tdd]"*)
            local stage
            stage="$(echo "$last_msg" | grep -oP 'stage \K[^:]+' || echo "?")"
            echo "tdd: $stage"
            ;;
        *"[ttnn-generic-op-builder]"*) echo "building stubs" ;;
        *"[ttnn-operation-architect]"*) echo "designing" ;;
        *"[ttnn-operation-analyzer]"*)  echo "analyzing" ;;
        *)
            if [[ "$build_done" == false ]]; then
                echo "building metal"
            else
                echo "running claude"
            fi
            ;;
    esac
}

# Runs in background. Polls all known runs every MONITOR_INTERVAL seconds.
# Args: prompt_names_csv num_runs session_dir results_dir starting_commit
monitor_progress() {
    local prompt_names_csv="$1"
    local num_runs="$2"
    local session_dir="$3"
    local results_dir="$4"
    local starting_commit="$5"
    local start_epoch
    start_epoch="$(date +%s)"

    IFS=',' read -ra prompt_names <<< "$prompt_names_csv"

    while true; do
        sleep "$MONITOR_INTERVAL"

        local now
        now="$(date +%s)"
        local total_elapsed=$(( now - start_epoch ))

        local active=0 done=0
        local lines=()

        for pname in "${prompt_names[@]}"; do
            for run in $(seq 1 "$num_runs"); do
                local label="${pname}:${run}"
                local clone_dir="${session_dir}/clones/${pname}_run${run}/tt-metal"
                local log_dir="${results_dir}/${pname}/run_${run}"

                # Check if this run finished (result.txt exists)
                if [[ -f "${log_dir}/result.txt" ]]; then
                    local result
                    result="$(cat "${log_dir}/result.txt")"
                    lines+=("$(printf "  %-25s DONE  |  %s" "[$label]" "$result")")
                    done=$((done + 1))
                    continue
                fi
                active=$((active + 1))

                local phase
                phase="$(detect_phase "$clone_dir")"

                # Count git commits beyond the starting branch
                local commits=0
                if [[ -d "$clone_dir/.git" ]]; then
                    commits="$(git -C "$clone_dir" rev-list --count HEAD ^"$starting_commit" 2>/dev/null || echo 0)"
                fi

                lines+=("$(printf "  %-25s ....  |  phase: %-20s  |  commits: %d" "[$label]" "$phase" "$commits")")
            done
        done

        echo ""
        local total=$(( active + done ))
        echo "--- Progress ($(format_elapsed $total_elapsed) elapsed) | active: ${active}, done: ${done}/${total} ---"
        for line in "${lines[@]}"; do
            echo "$line"
        done
        echo "---"

        # Stop if no active runs remain
        if [[ $active -eq 0 ]]; then
            break
        fi
    done
}

run_golden_tests() {
    # Run golden tests for a prompt inside the clone's subshell.
    # Args: $1=prompt_file $2=clone_dir $3=log_dir $4=prompt_name $5=run_id
    local prompt_file="$1" clone_dir="$2" log_dir="$3" prompt_name="$4" run_id="$5"

    # Parse "# golden: <op_name>" from prompt file
    local golden_op
    golden_op="$(grep -oP '^# golden: \K\S+' "$prompt_file" || true)"
    if [[ -z "$golden_op" ]]; then
        echo "[${prompt_name}:${run_id}] No golden test tag found, skipping"
        return 0
    fi

    local golden_dir="${clone_dir}/eval/golden_tests/${golden_op}"
    if [[ ! -d "$golden_dir" ]]; then
        echo "[${prompt_name}:${run_id}] Golden dir not found: ${golden_dir}, skipping"
        return 0
    fi

    echo "[${prompt_name}:${run_id}] Running golden tests for '${golden_op}'..."

    # Use eval_test_runner.sh for structured output (junitxml + classification)
    eval/eval_test_runner.sh "eval/golden_tests/${golden_op}/" "$log_dir" \
        > "${log_dir}/golden_test.log" 2>&1 || true

    # Read summary from structured output
    if [[ -f "${log_dir}/golden_results.txt" ]]; then
        local passed failed total
        passed="$(grep -oP 'PASSED=\K\d+' "${log_dir}/golden_results.txt" || echo 0)"
        failed="$(grep -oP 'FAILED=\K\d+' "${log_dir}/golden_results.txt" || echo 0)"
        total="$(grep -oP 'TOTAL=\K\d+' "${log_dir}/golden_results.txt" || echo 0)"
        echo "[${prompt_name}:${run_id}] Golden tests: ${passed}/${total} passed"
    else
        echo "[${prompt_name}:${run_id}] Golden tests: no results produced"
    fi
}

run_single() {
    local prompt_file="$1"
    local run_id="$2"
    local prompt_name
    prompt_name="$(basename "$prompt_file" .txt)"
    local clone_dir="${SESSION_DIR}/clones/${prompt_name}_run${run_id}/tt-metal"
    local log_dir="${RESULTS_DIR}/${prompt_name}/run_${run_id}"
    mkdir -p "$log_dir"

    local prompt_content
    prompt_content="$(cat "$prompt_file")"

    local start_time=$SECONDS

    echo "[${prompt_name}:${run_id}] Starting..."
    echo "[${prompt_name}:${run_id}]   Clone dir: ${clone_dir}"
    echo "[${prompt_name}:${run_id}]   Log dir:   ${log_dir}"
    echo "[${prompt_name}:${run_id}]   Prompt:    ${prompt_file}"

    # 1. Clone and create a unique branch from the source branch
    local run_branch="${DATE_STAMP}_${TIME_STAMP}_run${run_id}_${prompt_name}"
    echo "[${prompt_name}:${run_id}] Cloning ${BRANCH} -> ${run_branch}"
    git clone --branch "$BRANCH" "$REPO_URL" "$clone_dir" \
        > "${log_dir}/clone.log" 2>&1
    git -C "$clone_dir" checkout -b "$run_branch" \
        >> "${log_dir}/clone.log" 2>&1

    # 1b. Register SubagentStart hook for breadcrumb logging
    if [[ -f "${clone_dir}/.claude/scripts/logging/enable_breadcrumbs.sh" ]]; then
        (cd "$clone_dir" && .claude/scripts/logging/enable_breadcrumbs.sh) \
            >> "${log_dir}/clone.log" 2>&1
    fi

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
            echo "[${prompt_name}:${run_id}] Installing debugger..."
            # shellcheck disable=SC1091
            source python_env/bin/activate \
                && uv pip install -r tools/triage/requirements.txt >> "${log_dir}/venv.log" 2>&1 \
                && ./scripts/install_debugger.sh >> "${log_dir}/venv.log" 2>&1 || true
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

        # Wrap scripts/tt-test.sh: prepend a build gate so tests block until ready
        if [[ -f "${clone_dir}/scripts/tt-test.sh" ]]; then
            local original_test_sh
            original_test_sh="$(cat "${clone_dir}/scripts/tt-test.sh")"
            cat > "${clone_dir}/scripts/tt-test.sh" <<WRAPPER
#!/bin/bash
# Auto-generated wrapper: wait for build before running tests
source "${clone_dir}/.eval/wait_for_build.sh" || exit \$?
# --- Original scripts/tt-test.sh below ---
$(echo "$original_test_sh" | tail -n +2)
WRAPPER
            chmod +x "${clone_dir}/scripts/tt-test.sh"
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

        # 6b. Run golden tests (needs build+venv complete, device access via tt-test.sh)
        run_golden_tests "$prompt_file" "$clone_dir" "$log_dir" "$prompt_name" "$run_id"
    ) || claude_exit=$?

    echo "[${prompt_name}:${run_id}] Exited with code: $claude_exit"

    # 7. Record result
    local elapsed=$(( SECONDS - start_time ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))

    # Read golden results if available
    local golden_suffix=""
    if [[ -f "${log_dir}/golden_results.txt" ]]; then
        local g_passed g_total
        g_passed="$(grep -oP 'PASSED=\K\d+' "${log_dir}/golden_results.txt")"
        g_total="$(grep -oP 'TOTAL=\K\d+' "${log_dir}/golden_results.txt")"
        if [[ "$g_total" -gt 0 ]]; then
            golden_suffix=" (${g_passed}/${g_total} golden)"
        fi
    fi

    local result
    if [[ -f "${log_dir}/result.txt" ]] && grep -q "_FAIL" "${log_dir}/result.txt"; then
        result="$(cat "${log_dir}/result.txt")"
    elif [[ $claude_exit -eq 0 ]]; then
        result="PASS${golden_suffix:- (no golden tests)}"
    else
        result="FAIL (exit $claude_exit)${golden_suffix}"
    fi
    echo "$result" > "${log_dir}/result.txt"
    echo "[${prompt_name}:${run_id}] ${result} (${mins}m ${secs}s)"
    echo "${elapsed}" > "${log_dir}/duration_seconds.txt"

    # 8. Ingest into tracking database
    local ingest_args=(
        --prompt-name "$prompt_name"
        --run-number "$run_id"
        --starting-branch "$BRANCH"
        --starting-commit "$STARTING_COMMIT"
        --created-branch "$run_branch"
        --clone-dir "$clone_dir"
    )
    if [[ -f "${log_dir}/test_results.json" ]]; then
        ingest_args+=(--test-results "${log_dir}/test_results.json")
    fi
    # Score JSON is optional — only present if score.py was run
    if [[ -f "${log_dir}/score.json" ]]; then
        ingest_args+=(--score-json "${log_dir}/score.json")
    fi
    # Pass op name for kernel/artifact collection
    local golden_op_for_ingest
    golden_op_for_ingest="$(grep -oP '^# golden: \K\S+' "$prompt_file" || true)"
    if [[ -n "$golden_op_for_ingest" ]]; then
        ingest_args+=(--op-name "$golden_op_for_ingest")
        ingest_args+=(--golden-name "$golden_op_for_ingest")
    fi
    python3 -m eval.ingest "${ingest_args[@]}" \
        >> "${log_dir}/ingest.log" 2>&1 || \
        echo "[${prompt_name}:${run_id}] WARNING: DB ingest failed" >&2
}

# --- Build prompt names CSV for monitor ---
PROMPT_NAMES_CSV=""
for pf in "${PROMPT_FILES[@]}"; do
    pname="$(basename "$pf" .txt)"
    if [[ -n "$PROMPT_NAMES_CSV" ]]; then
        PROMPT_NAMES_CSV+=","
    fi
    PROMPT_NAMES_CSV+="$pname"
done

# --- Launch all runs in parallel ---
PIDS=()
for prompt_file in "${PROMPT_FILES[@]}"; do
    for run in $(seq 1 "$NUM_RUNS"); do
        run_single "$prompt_file" "$run" &
        PIDS+=($!)
    done
done

echo "Launched ${#PIDS[@]} parallel runs. Waiting for all to complete..."
echo "Progress updates every ${MONITOR_INTERVAL}s (--monitor-interval to change)"
echo ""

# Start background progress monitor
monitor_progress "$PROMPT_NAMES_CSV" "$NUM_RUNS" "$SESSION_DIR" "$RESULTS_DIR" "$STARTING_COMMIT" &
MONITOR_PID=$!

# Wait for all background jobs
for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

# Stop the progress monitor (it may have already exited on its own)
if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
fi

# --- Collect results from files ---
PASS_COUNT=0
FAIL_COUNT=0
ERROR_COUNT=0

while IFS= read -r result_file; do
    result="$(cat "$result_file")"
    case "$result" in
        PASS*)         PASS_COUNT=$((PASS_COUNT + 1)) ;;
        *_FAIL)        ERROR_COUNT=$((ERROR_COUNT + 1)) ;;
        *)             FAIL_COUNT=$((FAIL_COUNT + 1)) ;;
    esac
done < <(find "$RESULTS_DIR" -name "result.txt" -type f)

TOTAL=$((PASS_COUNT + FAIL_COUNT + ERROR_COUNT))

# --- Aggregate golden test results ---
GOLDEN_PASSED_TOTAL=0
GOLDEN_TESTS_TOTAL=0
GOLDEN_RUNS=0

while IFS= read -r golden_file; do
    local_passed="$(grep -oP 'PASSED=\K\d+' "$golden_file" || echo 0)"
    local_total="$(grep -oP 'TOTAL=\K\d+' "$golden_file" || echo 0)"
    if [[ "$local_total" -gt 0 ]]; then
        GOLDEN_PASSED_TOTAL=$((GOLDEN_PASSED_TOTAL + local_passed))
        GOLDEN_TESTS_TOTAL=$((GOLDEN_TESTS_TOTAL + local_total))
        GOLDEN_RUNS=$((GOLDEN_RUNS + 1))
    fi
done < <(find "$RESULTS_DIR" -name "golden_results.txt" -type f)

GOLDEN_SUMMARY=""
if [[ $GOLDEN_RUNS -gt 0 ]]; then
    GOLDEN_SUMMARY="
Golden:    ${GOLDEN_PASSED_TOTAL}/${GOLDEN_TESTS_TOTAL} passed (across ${GOLDEN_RUNS} run(s))"
fi

cat > "${RESULTS_DIR}/summary.txt" <<EOF
=== Evaluation Summary ===
Date:      $DATE_STAMP
Time:      $TIME_STAMP
Prompts:   ${#PROMPT_FILES[@]} file(s)
Branch:    $BRANCH
Runs:      $NUM_RUNS per prompt

Results:   $PASS_COUNT/$TOTAL passed
  PASS:    $PASS_COUNT
  FAIL:    $FAIL_COUNT
  INFRA:   $ERROR_COUNT${GOLDEN_SUMMARY}
===========================
EOF

echo ""
cat "${RESULTS_DIR}/summary.txt"

# --- Print session tree and clickable clone paths ---
echo ""
echo "=== Session Tree ==="
if command -v tree &>/dev/null; then
    tree -L 3 --dirsfirst "$SESSION_DIR"
else
    find "$SESSION_DIR" -maxdepth 3 -type d | sort | while IFS= read -r d; do
        depth=$(( $(echo "$d" | tr -cd '/' | wc -c) - $(echo "$SESSION_DIR" | tr -cd '/' | wc -c) ))
        indent=""
        for (( i=0; i<depth; i++ )); do indent+="  "; done
        echo "${indent}$(basename "$d")/"
    done
fi

echo ""
echo "=== Clone Repos ==="
find "${SESSION_DIR}/clones" -maxdepth 2 -name "tt-metal" -type d | sort | while IFS= read -r repo; do
    echo "  ${repo}"
done

# Exit non-zero if any run failed
[[ $FAIL_COUNT -eq 0 ]]
