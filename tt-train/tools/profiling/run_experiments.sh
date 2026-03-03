#!/bin/bash
#
# Profiling experiment runner (Slurm wrapper).
#
# Usage (single node):
#   ./run_experiments.sh --partition bh_pod_4x32_B45 --phases 1 2
#   ./run_experiments.sh --partition bh_pod_4x32_B45 --nodelist bh-glx-b08u02 --phases 1 2
#
# Usage (multiple nodes — distributes experiments round-robin):
#   ./run_experiments.sh --nodes bh_pod_4x32_B45:bh-glx-b04u02,bh_pod_4x32_B89:bh-glx-b08u02 --phases 1 2
#
# Usage (local — run directly on current node):
#   ./run_experiments.sh --local --phases 1 2
#
# Slurm args (parsed and removed before passing to python):
#   --partition <name>           Slurm partition (default: bh_pod_4x32_B45)
#   --nodelist <node>            Slurm node constraint (optional)
#   --nodes <p:n,p:n,...>        Multiple partition:node pairs (distributes experiments)
#   --local                      Run directly on current node (skip sbatch submission)
#
# All other args are forwarded to run_experiments.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ──
: "${TT_METAL_RUNTIME_ROOT:=/data/philei/tt-metal}"
PARTITION="bh_pod_4x32_B45"
NODELIST=""
NODES_MULTI=""
SLURM_TIME="12:00:00"
LOCAL_MODE=false
DRY_RUN=false

# ── Parse slurm-specific args, collect the rest for python ──
PYTHON_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --partition)  PARTITION="$2"; shift 2 ;;
        --nodelist)   NODELIST="$2"; shift 2 ;;
        --nodes)      NODES_MULTI="$2"; shift 2 ;;
        --local)      LOCAL_MODE=true; shift ;;
        --dry-run)    DRY_RUN=true; PYTHON_ARGS+=("$1"); shift ;;
        *)            PYTHON_ARGS+=("$1"); shift ;;
    esac
done

LOG_DIR="${TT_METAL_RUNTIME_ROOT}/tt-train/tools/profiling/slurm_logs"
mkdir -p "${LOG_DIR}"

PYTHON_SCRIPT="tt-train/tools/profiling/run_experiments.py"

# ── Multi-node: distribute experiments across nodes ──
if [ -n "${NODES_MULTI}" ] && [ -z "${SLURM_JOB_ID:-}" ] && [ "${LOCAL_MODE}" = false ]; then
    # Activate env to run python for --list-names
    source "${TT_METAL_RUNTIME_ROOT}/python_env/bin/activate"
    export PYTHONPATH="${TT_METAL_RUNTIME_ROOT}"

    # Get experiment names (respects --phases and other filters in PYTHON_ARGS)
    mapfile -t ALL_NAMES < <(
        cd "${TT_METAL_RUNTIME_ROOT}" && \
        python3 "${PYTHON_SCRIPT}" "${PYTHON_ARGS[@]}" --list-names
    )
    NUM_EXPS=${#ALL_NAMES[@]}

    # Parse partition:node pairs
    IFS=',' read -ra NODE_PAIRS <<< "${NODES_MULTI}"
    NUM_NODES=${#NODE_PAIRS[@]}

    echo "Distributing ${NUM_EXPS} experiments across ${NUM_NODES} nodes:"

    for idx in "${!NODE_PAIRS[@]}"; do
        IFS=':' read -r node_partition node_name <<< "${NODE_PAIRS[$idx]}"

        # Round-robin assignment
        NODE_EXPS=()
        for ((i = idx; i < NUM_EXPS; i += NUM_NODES)); do
            NODE_EXPS+=("${ALL_NAMES[$i]}")
        done

        if [ ${#NODE_EXPS[@]} -eq 0 ]; then
            echo "  Node ${node_name}: 0 experiments — skipping"
            continue
        fi

        echo "  Node ${node_name} (${node_partition}): ${#NODE_EXPS[@]} experiments"
        for exp_name in "${NODE_EXPS[@]}"; do
            echo "    - ${exp_name}"
        done

        if [ "${DRY_RUN}" = true ]; then
            continue
        fi

        SBATCH_ARGS=(
            --nodes=1
            --partition="${node_partition}"
            --nodelist="${node_name}"
            --job-name="profiling_${node_name}"
            --output="${LOG_DIR}/profiling_%j_${node_name}.out"
            --error="${LOG_DIR}/profiling_%j_${node_name}.err"
            --time="${SLURM_TIME}"
        )

        SCRIPT_ABS="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
        sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_ABS}" "${PYTHON_ARGS[@]}" --experiments "${NODE_EXPS[@]}"
    done

    if [ "${DRY_RUN}" = true ]; then
        echo ""
        echo "Dry run — no jobs submitted."
    fi

    exit 0
fi

# ── Single-node Slurm submission ──
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${LOCAL_MODE}" = false ]; then
    SBATCH_ARGS=(
        --nodes=1
        --partition="${PARTITION}"
        --job-name=train_profiling
        --output="${LOG_DIR}/profiling_%j.out"
        --error="${LOG_DIR}/profiling_%j.err"
        --time="${SLURM_TIME}"
    )
    [ -n "${NODELIST}" ] && SBATCH_ARGS+=(--nodelist="${NODELIST}")

    SCRIPT_ABS="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
    echo "Submitting to Slurm (partition=${PARTITION}, nodelist=${NODELIST:-any})..."
    sbatch "${SBATCH_ARGS[@]}" "${SCRIPT_ABS}" "${PYTHON_ARGS[@]}"
    exit $?
fi

# ── Running inside Slurm or --local ──
export TT_METAL_RUNTIME_ROOT
export TT_METAL_HOME="${TT_METAL_RUNTIME_ROOT}"
export PYTHONPATH="${TT_METAL_RUNTIME_ROOT}"
export TT_METAL_FABRIC_ROUTER_SYNC_TIMEOUT_MS=120000
export PYTHONUNBUFFERED=1
source "${TT_METAL_RUNTIME_ROOT}/python_env/bin/activate"

cd "${TT_METAL_RUNTIME_ROOT}"

echo "============================================"
echo "  Slurm Job ID : ${SLURM_JOB_ID:-local}"
echo "  Node          : $(hostname)"
echo "  Start time    : $(date)"
echo "  TT_METAL_HOME : ${TT_METAL_HOME}"
echo "============================================"

python3 "${PYTHON_SCRIPT}" "${PYTHON_ARGS[@]}"
exit_code=$?

echo "============================================"
echo "  End time      : $(date)"
echo "  Exit code     : ${exit_code}"
echo "============================================"

exit ${exit_code}
