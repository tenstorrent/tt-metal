#!/usr/bin/env bash
#SBATCH --job-name=sec3-galaxy
#SBATCH --partition=bh_sc5_B2B9_D12
#SBATCH --nodelist=bh-glx-b06u08
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --output=/data/rsong/tt-metal2/craq-parity-results/slurm-section3-galaxy-%j.out
#SBATCH --error=/data/rsong/tt-metal2/craq-parity-results/slurm-section3-galaxy-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${TT_METAL_HOME:-${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." REPO="${TT_METAL_HOME:-$(cd "${SCRIPT_DIR}/.." && pwd)}"REPO="${TT_METAL_HOME:-$(cd "${SCRIPT_DIR}/.." && pwd)}" pwd)}}"
# shellcheck source=scripts/lib/require-bh-glx-compute.sh
source "${REPO}/scripts/lib/require-bh-glx-compute.sh"
require_bh_glx_compute

export TT_METAL_HOME="$REPO"
export PYTHONPATH="${TT_METAL_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
export CRAQ_SIM="${CRAQ_SIM:-/data/rsong/craq-sim}"
export BUILD_DIR="${BUILD_DIR:-${REPO}/build_Debug}"
export SIM_ARCH=wh
export ARCH_NAME=wormhole_b0
export WH_MOCK_CLUSTER_DESC=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_cluster_desc.yaml
export PARITY_SECTIONS=3
export PER_CMD_TIMEOUT="${PER_CMD_TIMEOUT:-3600}"

STAMP="${RESULTS_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
export RESULTS_DIR="${RESULTS_DIR:-${REPO}/craq-parity-results/section3-galaxy-${STAMP}}"
mkdir -p "${RESULTS_DIR}"

if [ -f "${REPO}/python_env/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${REPO}/python_env/bin/activate"
fi

ln -sfn "$(basename "${BUILD_DIR}")" "${REPO}/build"
ln -sfn "$(basename "${RESULTS_DIR}")" "${REPO}/craq-parity-results/section3-galaxy-latest"

{
    echo "=== Section 3 Galaxy 32-chip WH sim ==="
    echo "=== Host: $(hostname -s) job=${SLURM_JOB_ID:-interactive} ==="
    echo "=== RESULTS_DIR=${RESULTS_DIR} mock=${WH_MOCK_CLUSTER_DESC} ==="
    echo "=== started_utc=$(date -u -Iseconds) ==="
} | tee "${RESULTS_DIR}/run.log"

exec >>"${RESULTS_DIR}/run.log" 2>&1
"${REPO}/scripts/run-nkapre-parity-ttsim-sweep.sh"
python3 "${REPO}/scripts/generate-parity-error-summary.py" "${RESULTS_DIR}" || true
echo "=== Done ==="
