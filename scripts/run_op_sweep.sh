#!/bin/bash
# run_op_sweep.sh — Generic op-sweep driver for ttnn unit tests.
#
# Wraps pytest with pytest-xdist work-stealing, optionally sweeping worker
# counts to measure how the simulator (or hardware) scales with parallelism.
# Works for any test file written against the standard ttnn `device` fixture.
#
# Usage:
#   scripts/run_op_sweep.sh <test_path> [-n N | --workers N] [pytest_args...]
#   scripts/run_op_sweep.sh <test_path> --scaling [worker_list] [pytest_args...]
#
# Examples:
#   # SDPA sweep, 4 workers, decode only, skip long-context heavies:
#   scripts/run_op_sweep.sh tests/ttnn/unit_tests/operations/sdpa/test_sdpa_sweep.py \
#       -n 4 -k decode -m "not slow"
#
#   # Conv2d scaling sweep at 1, 2, 4, 8 workers:
#   scripts/run_op_sweep.sh tests/ttnn/unit_tests/operations/conv/test_conv2d_sweep.py \
#       --scaling
#
#   # Layernorm scaling on a focused subset:
#   scripts/run_op_sweep.sh tests/ttnn/unit_tests/operations/fused/test_layer_norm_sweep.py \
#       --scaling "1 2 4" -k "llama or gpt"
#
# Env (auto-set; override before invoking to change):
#   TT_METAL_SIMULATOR   default: ~/sim/bh/libttsim.so  (sim mode auto-enabled if file exists)
#   ARCH_NAME            default: blackhole
#   TT_METAL_HOME        default: repo root
#
# Notes:
#   * --dist worksteal seeds each worker with an even slice, then idle workers
#     steal from busy ones — best when test durations are heterogeneous.
#   * Each xdist worker is a separate process that opens its own simulator
#     (or device) and creates its own ttnn device. Workers do not share state,
#     so on sim, scaling is bounded by host CPU cores.
#   * The first worker on a cold JIT cache pays the compile cost. Subsequent
#     workers benefit once the kernels are cached. For honest single-worker
#     numbers, drop $TT_METAL_CACHE before running.

set -o pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- First positional arg: test path ---
if [[ $# -lt 1 || "$1" == --help || "$1" == -h ]]; then
    sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 1
fi
TEST_PATH="$1"
shift

if [[ ! -e "$REPO_DIR/$TEST_PATH" && ! -e "$TEST_PATH" ]]; then
    echo "ERROR: test path not found: $TEST_PATH" >&2
    exit 1
fi

# --- Defaults ---
WORKERS=1
SCALING=false
SCALING_COUNTS="1 2 4 8"
PYTEST_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--workers)
            WORKERS="$2"
            shift 2
            ;;
        --scaling)
            SCALING=true
            if [[ -n "${2:-}" && "$2" != -* ]]; then
                SCALING_COUNTS="$2"
                shift
            fi
            shift
            ;;
        --)
            shift
            PYTEST_ARGS+=("$@")
            break
            ;;
        *)
            PYTEST_ARGS+=("$1")
            shift
            ;;
    esac
done

# --- Environment ---
export TT_METAL_SIMULATOR="${TT_METAL_SIMULATOR:-$HOME/sim/bh/libttsim.so}"
export ARCH_NAME="${ARCH_NAME:-blackhole}"
export TT_METAL_HOME="${TT_METAL_HOME:-$REPO_DIR}"

if [[ -f "$TT_METAL_SIMULATOR" ]]; then
    export TT_METAL_SLOW_DISPATCH_MODE=1
    export TT_METAL_DISABLE_SFPLOADMACRO=1
    SIM_NOTE="sim ($TT_METAL_SIMULATOR)"
else
    SIM_NOTE="hardware (TT_METAL_SIMULATOR=$TT_METAL_SIMULATOR not found, falling back)"
fi

cd "$REPO_DIR"
if [[ -f python_env/bin/activate ]]; then
    source python_env/bin/activate
fi

# --- Helpers ---
collected_count() {
    python -m pytest "$TEST_PATH" --collect-only -q -o addopts="" "${PYTEST_ARGS[@]}" 2>/dev/null \
        | grep -cE "::test_"
}

run_once() {
    local n="$1"
    local tag="$2"
    local logfile="$3"

    local start end elapsed
    start=$(date +%s)
    echo "[$(date '+%H:%M:%S')] [$tag] pytest -n $n --dist worksteal ${PYTEST_ARGS[*]}"
    pytest "$TEST_PATH" \
        -n "$n" --dist worksteal \
        -p no:cacheprovider \
        --timeout=1800 \
        -o addopts="--durations=0 --tb=short -rN" \
        "${PYTEST_ARGS[@]}" \
        > "$logfile" 2>&1
    local exit_code=$?
    end=$(date +%s)
    elapsed=$((end - start))

    local summary
    summary=$(grep -oE "[0-9]+ passed|[0-9]+ failed|[0-9]+ error|[0-9]+ skipped" "$logfile" | paste -sd, -)
    echo "[$(date '+%H:%M:%S')] [$tag] done in ${elapsed}s  exit=${exit_code}  ${summary:-no-summary}"
    # Emit machine-readable line on the last line
    echo "${n} ${elapsed} ${exit_code}"
}

# --- Banner ---
NUM_TESTS=$(collected_count || echo "?")
echo "================================================================"
echo "Op sweep harness"
echo "  test file       : $TEST_PATH"
echo "  collected tests : $NUM_TESTS"
echo "  backend         : $SIM_NOTE"
echo "  arch            : $ARCH_NAME"
echo "  pytest args     : ${PYTEST_ARGS[*]:-<none>}"
echo "================================================================"

if [[ "$SCALING" == true ]]; then
    LOG_DIR="${TMPDIR:-/tmp}/op_sweep_$$"
    mkdir -p "$LOG_DIR"
    echo "Scaling sweep across workers: $SCALING_COUNTS  (logs: $LOG_DIR)"
    echo

    declare -A ELAPSED
    BASELINE=""
    for n in $SCALING_COUNTS; do
        line=$(run_once "$n" "n=$n" "$LOG_DIR/n${n}.log" | tail -1)
        e=$(awk '{print $2}' <<< "$line")
        ELAPSED[$n]=$e
        if [[ -z "$BASELINE" ]]; then BASELINE=$e; fi
    done

    echo
    echo "================================================================"
    printf "%-8s %-10s %-12s %-12s\n" "workers" "elapsed_s" "speedup" "efficiency"
    echo "----------------------------------------------------------------"
    for n in $SCALING_COUNTS; do
        e=${ELAPSED[$n]}
        if [[ "$BASELINE" -gt 0 && "$e" -gt 0 ]]; then
            speedup=$(awk -v b="$BASELINE" -v e="$e" 'BEGIN{printf "%.2fx", b/e}')
            eff=$(awk -v b="$BASELINE" -v e="$e" -v n="$n" 'BEGIN{printf "%.0f%%", (b/e)/n*100}')
        else
            speedup="-"; eff="-"
        fi
        printf "%-8s %-10s %-12s %-12s\n" "$n" "$e" "$speedup" "$eff"
    done
    echo "================================================================"
    echo "Per-run logs: $LOG_DIR"
else
    LOG="${TMPDIR:-/tmp}/op_sweep_n${WORKERS}_$$.log"
    run_once "$WORKERS" "n=$WORKERS" "$LOG" >/dev/null
    grep -E "passed|failed|error|skipped" "$LOG" | tail -5
    echo "Log: $LOG"
fi
