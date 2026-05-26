#!/usr/bin/env bash
#SBATCH --job-name=sec2-quick
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=/data/rsong/tt-metal2/craq-parity-results/slurm-sec2-quick-%j.out
#SBATCH --error=/data/rsong/tt-metal2/craq-parity-results/slurm-sec2-quick-%j.err

set -euo pipefail

REPO="${TT_METAL_HOME:-${SLURM_SUBMIT_DIR:-/data/rsong/tt-metal2}}"
# shellcheck source=scripts/lib/require-bh-glx-compute.sh
source "${REPO}/scripts/lib/require-bh-glx-compute.sh"
require_bh_glx_compute

export TT_METAL_HOME="$REPO"
export PYTHONPATH="${TT_METAL_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
export CRAQ_SIM="${CRAQ_SIM:-/data/rsong/craq-sim}"
export BUILD_DIR="${BUILD_DIR:-${REPO}/build_Debug}"
export ARCH_NAME=wormhole_b0
export WH_MOCK_CLUSTER_DESC=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml
export PER_TEST_TIMEOUT="${PER_TEST_TIMEOUT:-600}"
unset TT_METAL_SLOW_DISPATCH_MODE

STAMP="${RESULTS_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
export RESULTS_DIR="${RESULTS_DIR:-${REPO}/craq-parity-results/section2-quick-${STAMP}}"
mkdir -p "${RESULTS_DIR}"

if [ -f "${REPO}/python_env/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${REPO}/python_env/bin/activate"
fi

ln -sfn "$(basename "${BUILD_DIR}")" "${REPO}/build"

WH_SIM_DIR="${RESULTS_DIR}/sim_wh_multichip"
mkdir -p "$WH_SIM_DIR"
cp "${CRAQ_SIM}/src/_out/release_wh/libttsim.so" "$WH_SIM_DIR/"
cp "${REPO}/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml" "$WH_SIM_DIR/soc_descriptor.yaml"

T3K_MOCK="$WH_MOCK_CLUSTER_DESC"
WH_ENV="ARCH_NAME=wormhole_b0 TT_METAL_SIMULATOR=$WH_SIM_DIR/libttsim.so TT_METAL_SIMULATOR_HOME=$WH_SIM_DIR TT_METAL_MOCK_CLUSTER_DESC_PATH=$T3K_MOCK TT_METAL_DRAM_BACKED_CQ=1 TT_METAL_SIMULATOR_CQ_WAIT_CLOCKS=10000"
MOCK_ENV="env -u TT_METAL_SIMULATOR -u TT_METAL_SIMULATOR_HOME ARCH_NAME=wormhole_b0 TT_METAL_MOCK_CLUSTER_DESC_PATH=$T3K_MOCK TT_METAL_SLOW_DISPATCH_MODE=1"
export TT_METAL_DISABLE_SFPLOADMACRO=1
export LD_LIBRARY_PATH="${BUILD_DIR}/tt_metal:${BUILD_DIR}/tt_metal/third_party/umd/lib:${BUILD_DIR}/ttnn:${BUILD_DIR}/tt_stl:${BUILD_DIR}/lib:${LD_LIBRARY_PATH:-}"

SUMMARY="${RESULTS_DIR}/summary.tsv"
RUN_LOG="${RESULTS_DIR}/run.log"
: >"$SUMMARY"
echo -e "suite\tstatus\trc\tduration_sec" >>"$SUMMARY"

{
    echo "=== Section 2 quick verify (no rebuild) ==="
    echo "=== Host: $(hostname -s) job=${SLURM_JOB_ID:-interactive} ==="
    echo "=== craq-sim=$(git -C "$CRAQ_SIM" log -1 --oneline) ==="
} | tee "$RUN_LOG"

run_one() {
    local name="$1"
    shift
    local log="${RESULTS_DIR}/${name}.log"
    echo "=== RUN $name ===" | tee -a "$RUN_LOG"
    local start end dur rc status
    start=$(date +%s)
    set +e
    timeout "$PER_TEST_TIMEOUT" bash -lc "cd '$REPO' && $*" >"$log" 2>&1
    rc=$?
    set -e
    end=$(date +%s)
    dur=$((end - start))
    case "$rc" in
        0) status=PASS ;;
        124) status=TIMEOUT ;;
        *) status=FAIL ;;
    esac
    echo "$name $status rc=$rc ${dur}s" | tee -a "$RUN_LOG"
    echo -e "${name}\t${status}\t${rc}\t${dur}" >>"$SUMMARY"
    tail -5 "$log" >>"$RUN_LOG" || true
}

# Easiest / shortest failures from 11604 baseline (fixes already in tree).
run_one "run_visible_devices_mp" \
    "$WH_ENV ./tests/tt_metal/distributed/multiprocess/run_visible_devices_mp_tests.sh"

run_one "fabric_control_T3k_mock" \
    "$MOCK_ENV ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter='ControlPlaneFixture.*T3k*'"

run_one "eth_direct_send" \
    "$WH_ENV TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_eth --gtest_filter='MeshDeviceFixture.ActiveEthKernelsDirectSendAllConnectedChips'"

run_one "fabric_unicast_1x8_wh_sim" \
    "$WH_ENV TT_METAL_SLOW_DISPATCH_MODE=1 TT_MESH_GRAPH_DESC_PATH=tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_1x8_mesh_graph_descriptor.textproto ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter='*Fabric2DFixture.TestUnicast*'"

run_one "fabric_worker_edm" \
    "$WH_ENV ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter='*WorkerFabricEdmDatapath*:*EdmFabric*'"

echo "=== Done ===" | tee -a "$RUN_LOG"
cat "$SUMMARY" | tee -a "$RUN_LOG"
