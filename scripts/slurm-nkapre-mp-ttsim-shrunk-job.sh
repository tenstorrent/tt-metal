#!/usr/bin/env bash
#SBATCH --job-name=nkapre-mp-shrunk
#SBATCH --partition=bh_sc5_B2B9_D12
#SBATCH --nodelist=bh-glx-b06u08
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --output=/data/rsong/tt-metal2/craq-parity-results/slurm-mp-shrunk-%j.out
#SBATCH --error=/data/rsong/tt-metal2/craq-parity-results/slurm-mp-shrunk-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${TT_METAL_HOME:-${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}}"
# shellcheck source=scripts/lib/require-bh-glx-compute.sh
source "${REPO}/scripts/lib/require-bh-glx-compute.sh"
require_bh_glx_compute

export TT_METAL_HOME="$REPO"
export BUILD_DIR="${REPO}/build_Debug"
export CRAQ_SIM="${CRAQ_SIM:-/data/rsong/craq-sim}"
export TTSIM_USE_DAEMON="${TTSIM_USE_DAEMON:-auto}"
export PER_CMD_TIMEOUT="${PER_CMD_TIMEOUT:-1800}"
unset TT_METAL_SIMULATOR_HOME TT_METAL_SIMULATOR TT_METAL_MOCK_CLUSTER_DESC_PATH
STAMP="${RESULTS_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
export RESULTS_DIR="${RESULTS_DIR:-${REPO}/craq-parity-results/mp-run-shrunk-${STAMP}}"

if [ -f "${REPO}/python_env/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${REPO}/python_env/bin/activate"
fi

mkdir -p "${RESULTS_DIR}"
ln -sfn "$(basename "${RESULTS_DIR}")" "${REPO}/craq-parity-results/mp-latest-shrunk"

echo "=== Compute node: $(hostname -s) ==="
echo "=== RESULTS_DIR=${RESULTS_DIR} ==="
echo "=== PER_CMD_TIMEOUT=${PER_CMD_TIMEOUT}s ==="
echo "=== Tail: tail -f ${RESULTS_DIR}/run.log ==="

cd "$REPO"
chmod +x scripts/run-nkapre-mp-ttsim-sweep-shrunk.sh
exec scripts/run-nkapre-mp-ttsim-sweep-shrunk.sh
