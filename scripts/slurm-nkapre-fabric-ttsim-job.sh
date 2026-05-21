#!/usr/bin/env bash
#SBATCH --job-name=fabric-ttsim
#SBATCH --partition=bh_sc5_B2B9_D12
#SBATCH --nodelist=bh-glx-b06u08
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=/data/rsong/tt-metal2/craq-parity-results/slurm-fabric-%j.out
#SBATCH --error=/data/rsong/tt-metal2/craq-parity-results/slurm-fabric-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${TT_METAL_HOME:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
# shellcheck source=scripts/lib/require-bh-glx-compute.sh
source "${REPO}/scripts/lib/require-bh-glx-compute.sh"
require_bh_glx_compute

export TT_METAL_HOME="$REPO"
export PYTHONPATH="${TT_METAL_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
export CRAQ_SIM="${CRAQ_SIM:-/data/rsong/craq-sim}"
export BUILD_DIR="${BUILD_DIR:-${REPO}/build_Debug}"
export SIM_ARCH=wh
export ARCH_NAME=wormhole_b0
export WH_MOCK_CLUSTER_DESC=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml
export PARITY_SECTIONS=2.fabric
export PARITY_INCLUDE_MP=0
export PER_CMD_TIMEOUT="${PER_CMD_TIMEOUT:-3600}"
unset TT_METAL_SLOW_DISPATCH_MODE

STAMP="${RESULTS_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
export RESULTS_DIR="${RESULTS_DIR:-${REPO}/craq-parity-results/fabric-${STAMP}}"
mkdir -p "${RESULTS_DIR}"

if [ -f "${REPO}/python_env/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${REPO}/python_env/bin/activate"
fi

ln -sfn "$(basename "${BUILD_DIR}")" "${REPO}/build"
ln -sfn "$(basename "${RESULTS_DIR}")" "${REPO}/craq-parity-results/fabric-latest"

{
    echo "=== Fabric ttsim sweep (WH multichip + T3K mock) ==="
    echo "=== Host: $(hostname -s) job=${SLURM_JOB_ID:-interactive} ==="
    echo "=== RESULTS_DIR=${RESULTS_DIR} ==="
    echo "=== CRAQ_SIM=${CRAQ_SIM} ==="
    echo "=== started_utc=$(date -u -Iseconds) ==="
} | tee "${RESULTS_DIR}/run.log"

exec >>"${RESULTS_DIR}/run.log" 2>&1
"${REPO}/scripts/run-nkapre-parity-ttsim-sweep.sh"

python3 "${REPO}/scripts/generate-parity-error-summary.py" "${RESULTS_DIR}" || true

echo "=== Done ==="
echo "Summary: ${RESULTS_DIR}/summary.tsv"
