#!/usr/bin/env bash
#SBATCH --job-name=sec1-cpp
#SBATCH --partition=bh_sc5_B2B9_D12
#SBATCH --nodelist=bh-glx-b06u08
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=/data/rsong/tt-metal2/craq-parity-results/slurm-section1-cpp-%j.out
#SBATCH --error=/data/rsong/tt-metal2/craq-parity-results/slurm-section1-cpp-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${TT_METAL_HOME:-${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}}"
# shellcheck source=scripts/lib/require-bh-glx-compute.sh
source "${REPO}/scripts/lib/require-bh-glx-compute.sh"
require_bh_glx_compute

export TT_METAL_HOME="$REPO"
export PYTHONPATH="${TT_METAL_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
export CRAQ_SIM="${CRAQ_SIM:-/data/rsong/craq-sim}"
export BUILD_DIR="${BUILD_DIR:-${REPO}/build_Debug}"
export SIM_ARCH=bh
export ARCH_NAME=blackhole
export PARITY_SECTIONS=1
export PARITY_SKIP_PYTEST=1
export PER_CMD_TIMEOUT="${PER_CMD_TIMEOUT:-3600}"
unset TT_METAL_SLOW_DISPATCH_MODE

STAMP="${RESULTS_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
export RESULTS_DIR="${RESULTS_DIR:-${REPO}/craq-parity-results/section1-cpp-${STAMP}}"
mkdir -p "${RESULTS_DIR}"

if [ -f "${REPO}/python_env/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${REPO}/python_env/bin/activate"
fi

ln -sfn "$(basename "${BUILD_DIR}")" "${REPO}/build"
ln -sfn "$(basename "${RESULTS_DIR}")" "${REPO}/craq-parity-results/section1-cpp-latest"

{
    echo "=== Section 1 C++ gtests (BH ttsim, craq-sim $(git -C "$CRAQ_SIM" log -1 --oneline)) ==="
    echo "=== RESULTS_DIR=${RESULTS_DIR} ==="
    echo "=== started_utc=$(date -u -Iseconds) ==="
} | tee "${RESULTS_DIR}/run.log"

exec >>"${RESULTS_DIR}/run.log" 2>&1
"${REPO}/scripts/run-nkapre-parity-ttsim-sweep.sh"
python3 "${REPO}/scripts/generate-parity-error-summary.py" "${RESULTS_DIR}" || true
echo "=== Done ==="
