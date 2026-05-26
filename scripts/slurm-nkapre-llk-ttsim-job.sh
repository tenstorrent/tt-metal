#!/usr/bin/env bash
#SBATCH --job-name=nkapre-llk
#SBATCH --partition=bh_sc5_B2B9_D12
#SBATCH --nodelist=bh-glx-b06u08
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=/data/rsong/tt-metal2/craq-parity-results/slurm-llk-%j.out
#SBATCH --error=/data/rsong/tt-metal2/craq-parity-results/slurm-llk-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${TT_METAL_HOME:-${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}}"
# shellcheck source=scripts/lib/require-bh-glx-compute.sh
source "${REPO}/scripts/lib/require-bh-glx-compute.sh"
require_bh_glx_compute

export TT_METAL_HOME="$REPO"
export CRAQ_SIM="${CRAQ_SIM:-/data/rsong/craq-sim}"
STAMP="${RESULTS_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
export RESULTS_DIR="${RESULTS_DIR:-${REPO}/craq-parity-results/llk-run-${STAMP}}"

if [ -f "${REPO}/python_env/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${REPO}/python_env/bin/activate"
fi

mkdir -p "${RESULTS_DIR}"
ln -sfn "$(basename "${RESULTS_DIR}")" "${REPO}/craq-parity-results/llk-latest"

echo "=== Compute node: $(hostname -s) ==="
echo "=== RESULTS_DIR=${RESULTS_DIR} ==="
echo "=== Tail: tail -f ${RESULTS_DIR}/run.log ==="

cd "$REPO"
chmod +x scripts/run-nkapre-llk-ttsim-sweep.sh scripts/setup-llk-ttsim-env.sh
exec > >(tee -a "${RESULTS_DIR}/run.log") 2>&1
exec scripts/run-nkapre-llk-ttsim-sweep.sh
