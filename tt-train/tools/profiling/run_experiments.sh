#!/bin/bash
#
# Profiling experiment runner (Slurm wrapper).
#
# Usage (submits to Slurm automatically):
#   ./run_experiments.sh --phases 1 2
#   ./run_experiments.sh --partition bh_pod_4x32_B89 --nodelist bh-glx-b08u02 --phases 1 2
#   ./run_experiments.sh --partition bh_pod_4x32_B45 --name tinyllama --max-steps 12
#
# Slurm args (parsed and removed before passing to python):
#   --partition <name>    Slurm partition (default: bh_pod_4x32_B45)
#   --nodelist <node>     Slurm node constraint (optional)
#   --local                Run directly on current node (skip sbatch submission)
#
# All other args are forwarded to run_experiments.py.

set -euo pipefail

# ── Defaults ──
: "${TT_METAL_RUNTIME_ROOT:=/data/philei/tt-metal}"
PARTITION="bh_pod_4x32_B45"
NODELIST=""
SLURM_TIME="12:00:00"
LOCAL_MODE=false

# ── Parse slurm-specific args, collect the rest for python ──
PYTHON_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --partition)  PARTITION="$2"; shift 2 ;;
        --nodelist)   NODELIST="$2"; shift 2 ;;
        --local)      LOCAL_MODE=true; shift ;;
        *)            PYTHON_ARGS+=("$1"); shift ;;
    esac
done

LOG_DIR="${TT_METAL_RUNTIME_ROOT}/tt-train/tools/profiling/slurm_logs"
mkdir -p "${LOG_DIR}"

# ── If not inside Slurm and not --local, submit ourselves ──
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

    echo "Submitting to Slurm (partition=${PARTITION}, nodelist=${NODELIST:-any})..."
    sbatch "${SBATCH_ARGS[@]}" "$0" "${PYTHON_ARGS[@]}"
    exit $?
fi

# ── Running inside Slurm ──
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

python3 tt-train/tools/profiling/run_experiments.py "${PYTHON_ARGS[@]}"
exit_code=$?

echo "============================================"
echo "  End time      : $(date)"
echo "  Exit code     : ${exit_code}"
echo "============================================"

exit ${exit_code}
