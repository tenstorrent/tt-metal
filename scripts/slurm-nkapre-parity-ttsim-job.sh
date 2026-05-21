#!/usr/bin/env bash
#SBATCH --job-name=nkapre-ttsim
#SBATCH --partition=bh_sc5_B2B9_D12
#SBATCH --nodelist=bh-glx-b06u08
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=/data/rsong/tt-metal2/craq-parity-results/slurm-%j.out
#SBATCH --error=/data/rsong/tt-metal2/craq-parity-results/slurm-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${TT_METAL_HOME:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
# shellcheck source=scripts/lib/require-bh-glx-compute.sh
source "${REPO}/scripts/lib/require-bh-glx-compute.sh"
require_bh_glx_compute

CRAQ=/data/rsong/craq-sim
export TT_METAL_HOME="$REPO"
export ARCH_NAME=blackhole
export SIM_ARCH=bh
export CRAQ_SIM="${CRAQ}"
export PYTHONPATH="${TT_METAL_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
export PER_CMD_TIMEOUT="${PER_CMD_TIMEOUT:-900}"

STAMP="${RESULTS_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
export RESULTS_DIR="${RESULTS_DIR:-${REPO}/craq-parity-results/run-${STAMP}}"
mkdir -p "${RESULTS_DIR}"

if [ -f "${REPO}/python_env/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${REPO}/python_env/bin/activate"
fi

NINJA_J="${NINJA_J:-${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}}"
if [ "${NINJA_J}" -gt 32 ]; then
    NINJA_J=32
fi
export CMAKE_BUILD_PARALLEL_LEVEL="${NINJA_J}"

NODE="$(hostname -s)"
RUN_LOG="${RESULTS_DIR}/run.log"
LATEST_LINK="${REPO}/craq-parity-results/latest"

ln -sfn "$(basename "${RESULTS_DIR}")" "${LATEST_LINK}"

{
    echo "=== Compute node: ${NODE} ==="
    echo "=== Date: $(date -u) ==="
    echo "=== Mode: craq-sim ttsim (no hardware) ==="
    echo "=== CRAQ_SIM=${CRAQ} ==="
    echo "=== RESULTS_DIR=${RESULTS_DIR} ==="
    echo "=== Tail this log: tail -f ${RUN_LOG} ==="
    echo "=== Slurm job: ${SLURM_JOB_ID:-interactive} ==="
} | tee "${RUN_LOG}"

cd "$REPO"
chmod +x scripts/run-nkapre-parity-ttsim-full.sh scripts/run-nkapre-parity-ttsim-sweep.sh
chmod +x scripts/generate-parity-error-summary.py

export BUILD_DIR="${REPO}/build_Debug"
export RESULTS_STAMP="${STAMP}"

# Stream build + sweep into run.log (also mirrored to slurm-%j.out via tee below).
exec > >(tee -a "${RUN_LOG}") 2>&1
"${REPO}/scripts/run-nkapre-parity-ttsim-full.sh"

echo "=== Generating ERRORS.md ==="
python3 "${REPO}/scripts/generate-parity-error-summary.py" "${RESULTS_DIR}"

echo "=== Done ==="
echo "Summary TSV: ${RESULTS_DIR}/summary.tsv"
echo "Error report: ${RESULTS_DIR}/ERRORS.md"
echo "Live log: tail -f ${RUN_LOG}"
