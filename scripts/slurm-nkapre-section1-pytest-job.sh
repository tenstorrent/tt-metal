#!/usr/bin/env bash
#SBATCH --job-name=sec1-pytest
#SBATCH --partition=bh_sc5_B2B9_D12
#SBATCH --nodelist=bh-glx-b06u08
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=/data/rsong/tt-metal2/craq-parity-results/slurm-section1-pytest-%j.out
#SBATCH --error=/data/rsong/tt-metal2/craq-parity-results/slurm-section1-pytest-%j.err

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
export SIM_ARCH=bh
export ARCH_NAME=blackhole
export PARITY_SECTIONS=1
export PER_CMD_TIMEOUT="${PER_CMD_TIMEOUT:-3600}"
unset TT_METAL_SLOW_DISPATCH_MODE

STAMP="${RESULTS_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
export RESULTS_DIR="${RESULTS_DIR:-${REPO}/craq-parity-results/section1-pytest-${STAMP}}"
mkdir -p "${RESULTS_DIR}"

if [ -f "${REPO}/python_env/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${REPO}/python_env/bin/activate"
fi

ln -sfn "$(basename "${BUILD_DIR}")" "${REPO}/build"
ln -sfn "$(basename "${RESULTS_DIR}")" "${REPO}/craq-parity-results/section1-pytest-latest"

{
    echo "=== Section 1 pytest re-run (fast dispatch + thread timeout) ==="
    echo "=== Host: $(hostname -s) job=${SLURM_JOB_ID:-interactive} ==="
    echo "=== RESULTS_DIR=${RESULTS_DIR} ==="
    echo "=== started_utc=$(date -u -Iseconds) ==="
} | tee "${RESULTS_DIR}/run.log"

exec >>"${RESULTS_DIR}/run.log" 2>&1

PYTHON="${REPO}/python_env/bin/python3"
SIM_DIR="${RESULTS_DIR}/sim"
mkdir -p "$SIM_DIR"
cp "${CRAQ_SIM}/src/_out/release_bh/libttsim.so" "$SIM_DIR/"
cp "${REPO}/tt_metal/soc_descriptors/blackhole_140_arch.yaml" "$SIM_DIR/soc_descriptor.yaml"

export TT_METAL_SIMULATOR="$SIM_DIR/libttsim.so"
export TT_METAL_SIMULATOR_HOME="$SIM_DIR"
export TT_METAL_DISABLE_SFPLOADMACRO=1
export TT_METAL_DRAM_BACKED_CQ=1
export TT_METAL_SIMULATOR_CQ_WAIT_CLOCKS=10000
export LD_LIBRARY_PATH="${BUILD_DIR}/tt_metal:${BUILD_DIR}/tt_metal/third_party/umd/lib:${BUILD_DIR}/ttnn:${BUILD_DIR}/tt_stl:${BUILD_DIR}/lib:${LD_LIBRARY_PATH:-}"

cd "$REPO"
echo "=== pytest tests/ttnn/unit_tests/ (fast dispatch, thread timeout) ==="
set +e
env -u TT_METAL_MOCK_CLUSTER_DESC_PATH -u TT_METAL_SLOW_DISPATCH_MODE \
    timeout "${PER_CMD_TIMEOUT}" \
    "$PYTHON" -m pytest tests/ttnn/unit_tests/ -xvvv --timeout-method=thread \
    2>&1 | tee "${RESULTS_DIR}/pytest.log"
rc=${PIPESTATUS[0]}
set -e
echo "pytest exit=$rc" | tee "${RESULTS_DIR}/summary.txt"
echo -e "1.ttnn_py/unit_tests\t$([ "$rc" -eq 0 ] && echo PASS || echo FAIL)\t${rc}\t0\tpytest fast dispatch" >"${RESULTS_DIR}/summary.tsv"

python3 "${REPO}/scripts/generate-parity-error-summary.py" "${RESULTS_DIR}" || true
echo "=== Done ==="
