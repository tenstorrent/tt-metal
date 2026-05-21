#!/usr/bin/env bash
# Run all commands from nkapre-fork-test-commands.md via craq-sim ttsim.
# Writes summary.tsv, per-suite logs, tail-able run.log, and ERRORS.md at end.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/require-bh-glx-compute.sh
source "${REPO_ROOT}/scripts/lib/require-bh-glx-compute.sh"
require_bh_glx_compute

CRAQ_SIM="${CRAQ_SIM:-/data/rsong/craq-sim}"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build_Debug}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/craq-parity-results/run-$(date -u +%Y%m%dT%H%M%SZ)}"
ARCH="${ARCH_NAME:-blackhole}"
PER_CMD_TIMEOUT="${PER_CMD_TIMEOUT:-900}"
SIM_ARCH="${SIM_ARCH:-bh}"
PARITY_SECTIONS="${PARITY_SECTIONS:-1,2,3,4}"
PARITY_INCLUDE_MP="${PARITY_INCLUDE_MP:-1}"

section_enabled() {
    local n="$1"
    [[ ",${PARITY_SECTIONS}," == *",${n},"* ]]
}

mkdir -p "$RESULTS_DIR"
SUMMARY="$RESULTS_DIR/summary.tsv"
RUN_LOG="$RESULTS_DIR/run.log"
: >"$SUMMARY"
: >"$RUN_LOG"
echo -e "suite\tstatus\texit_code\tduration_sec\tcommand" >>"$SUMMARY"

export TT_METAL_HOME="$REPO_ROOT"
export PYTHONPATH="${TT_METAL_HOME}${PYTHONPATH:+:$PYTHONPATH}"
export ARCH_NAME="$ARCH"
export TT_METAL_SLOW_DISPATCH_MODE="${TT_METAL_SLOW_DISPATCH_MODE:-1}"
export TT_METAL_DISABLE_SFPLOADMACRO=1
export TT_METAL_DRAM_BACKED_CQ=1
export TT_METAL_SIMULATOR_CQ_WAIT_CLOCKS=10000
export LD_LIBRARY_PATH="${BUILD_DIR}/tt_metal:${BUILD_DIR}/tt_metal/third_party/umd/lib:${BUILD_DIR}/ttnn:${BUILD_DIR}/tt_stl:${BUILD_DIR}/lib:${LD_LIBRARY_PATH:-}"

PYTHON="${REPO_ROOT}/python_env/bin/python3"
if [ ! -x "$PYTHON" ]; then
    PYTHON="python3"
fi

# craq-sim LLK harness expects tt_metal/tt-llk/tests/.venv and sfpi under the same tree.
setup_llk_env() {
    if [ -x "$REPO_ROOT/scripts/setup-llk-ttsim-env.sh" ]; then
        "$REPO_ROOT/scripts/setup-llk-ttsim-env.sh"
    fi
}

case "$SIM_ARCH" in
    wh|wormhole*)
        SIM_SRC="$CRAQ_SIM/src/_out/release_wh/libttsim.so"
        SOC="$TT_METAL_HOME/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml"
        ;;
    bh|blackhole*)
        SIM_SRC="$CRAQ_SIM/src/_out/release_bh/libttsim.so"
        SOC="$TT_METAL_HOME/tt_metal/soc_descriptors/blackhole_140_arch.yaml"
        ;;
    *)
        echo "ERROR: unknown SIM_ARCH=$SIM_ARCH" >&2
        exit 2
        ;;
esac

SIM_DIR="$RESULTS_DIR/sim"
mkdir -p "$SIM_DIR"
cp "$SIM_SRC" "$SIM_DIR/libttsim.so"
cp "$SOC" "$SIM_DIR/soc_descriptor.yaml"
export TT_METAL_SIMULATOR="$SIM_DIR/libttsim.so"
export TT_METAL_SIMULATOR_HOME="$SIM_DIR"

# Mock cluster descriptors for multichip ttsim (required; hardware descriptors fail on sim).
T3K_MOCK="tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml"
P300_MOCK="tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_P300_both_mmio.yaml"
WH6U_MOCK="tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/6u_cluster_desc.yaml"

BH_SIM_DIR="$RESULTS_DIR/sim_bh_multichip"
mkdir -p "$BH_SIM_DIR"
cp "$CRAQ_SIM/src/_out/release_bh/libttsim.so" "$BH_SIM_DIR/libttsim.so"
cp "$TT_METAL_HOME/tt_metal/soc_descriptors/blackhole_140_arch.yaml" "$BH_SIM_DIR/soc_descriptor.yaml"

WH_SIM_DIR="$RESULTS_DIR/sim_wh_multichip"
mkdir -p "$WH_SIM_DIR"
cp "$CRAQ_SIM/src/_out/release_wh/libttsim.so" "$WH_SIM_DIR/libttsim.so"
cp "$TT_METAL_HOME/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml" "$WH_SIM_DIR/soc_descriptor.yaml"

MOCK_FABRIC_ENV="env -u TT_METAL_SIMULATOR -u TT_METAL_SIMULATOR_HOME ARCH_NAME=wormhole_b0 TT_METAL_MOCK_CLUSTER_DESC_PATH=$T3K_MOCK TT_METAL_SLOW_DISPATCH_MODE=1"
BH_MULTICHIP_ENV="ARCH_NAME=blackhole TT_METAL_SIMULATOR=$BH_SIM_DIR/libttsim.so TT_METAL_SIMULATOR_HOME=$BH_SIM_DIR TT_METAL_MOCK_CLUSTER_DESC_PATH=$P300_MOCK TT_METAL_DRAM_BACKED_CQ=1 TT_METAL_SIMULATOR_CQ_WAIT_CLOCKS=10000"
WH_MULTICHIP_ENV="ARCH_NAME=wormhole_b0 TT_METAL_SIMULATOR=$WH_SIM_DIR/libttsim.so TT_METAL_SIMULATOR_HOME=$WH_SIM_DIR TT_METAL_MOCK_CLUSTER_DESC_PATH=${WH_MOCK_CLUSTER_DESC:-$T3K_MOCK} TT_METAL_DRAM_BACKED_CQ=1 TT_METAL_SIMULATOR_CQ_WAIT_CLOCKS=10000"

if [ ! -e "$REPO_ROOT/build" ] || [ -L "$REPO_ROOT/build" ]; then
    ln -sfn "$(basename "$BUILD_DIR")" "$REPO_ROOT/build"
fi

log() {
    local msg="[sweep] $*"
    echo "$msg" >>"$RUN_LOG"
    echo "$msg" >>"$RESULTS_DIR/sweep.log"
}

run_cmd() {
    local suite="$1"
    shift
    local cmd=("$@")
    local logfile="$RESULTS_DIR/$(echo "$suite" | tr '/ ' '__').log"
    local start end dur rc status

    log "RUN [$suite]: ${cmd[*]}"
    {
        echo ""
        echo "===== RUN [$suite] $(date -u -Iseconds) ====="
        echo "CMD: ${cmd[*]}"
    } >>"$RUN_LOG"

    start=$(date +%s)
    set +e
    timeout "$PER_CMD_TIMEOUT" bash -lc "cd '$REPO_ROOT' && ${cmd[*]}" 2>&1 | tee -a "$RUN_LOG" | tee "$logfile"
    rc=${PIPESTATUS[0]}
    set -e
    end=$(date +%s)
    dur=$((end - start))

    case "$rc" in
        0) status=PASS ;;
        124) status=TIMEOUT ;;
        127) status=MISSING ;;
        *) status=FAIL ;;
    esac

    {
        echo "===== DONE [$suite] status=$status rc=$rc duration=${dur}s ====="
    } >>"$RUN_LOG"

    echo -e "${suite}\t${status}\t${rc}\t${dur}\t${cmd[*]}" >>"$SUMMARY"
    log "DONE [$suite]: $status (rc=$rc, ${dur}s)"
}

if section_enabled 1; then
# --- Section 1: TTNN Tests (single card) ---
# Do not inherit WH mock cluster desc from Galaxy login/slurm env; it breaks BH ttsim init.
for t in unit_tests_ttnn unit_tests_ttnn_tensor unit_tests_ttnn_ccl \
    unit_tests_ttnn_ccl_multi_tensor unit_tests_ttnn_ccl_ops unit_tests_ttnn_accessor \
    test_ccl_multi_cq_multi_device; do
    run_cmd "1.ttnn_cpp/$t" "env -u TT_METAL_MOCK_CLUSTER_DESC_PATH ./build/test/ttnn/$t"
done

run_cmd "1.ttnn_py/unit_tests" "env -u TT_METAL_MOCK_CLUSTER_DESC_PATH $PYTHON -m pytest tests/ttnn/unit_tests/ -xvvv --timeout-method=thread"

fi

if section_enabled 2; then
# --- Section 2: T3000 Tests (single-host; multiprocess tt-run gated by PARITY_INCLUDE_MP) ---
run_cmd "2.distributed/distributed_unit_tests" \
    "$WH_MULTICHIP_ENV ./build/test/tt_metal/distributed/distributed_unit_tests"
run_cmd "2.distributed/run_visible_devices_mp" \
    "$WH_MULTICHIP_ENV ./tests/tt_metal/distributed/multiprocess/run_visible_devices_mp_tests.sh"

for filt in \
    "MeshDeviceFixture.ActiveEthKernelsDirectSendAllConnectedChips" \
    "MeshDeviceFixture.ActiveEthKernelsSendInterleavedBufferAllConnectedChips" \
    "MeshDeviceFixture.ActiveEthKernelsDirectRingGatherAllChips" \
    "MeshDeviceFixture.ActiveEthKernelsInterleavedRingGatherAllChips"; do
    run_cmd "2.eth/$filt" \
        "$WH_MULTICHIP_ENV TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_eth --gtest_filter=$filt"
done

run_cmd "2.dispatch/CommandQueueSingleCard" \
    "$WH_MULTICHIP_ENV TT_METAL_ENABLE_REMOTE_CHIP=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter='CommandQueueSingleCard*Fixture.*'"
run_cmd "2.dispatch/CommandQueueMultiDevice" \
    "$WH_MULTICHIP_ENV ./build/test/tt_metal/unit_tests_dispatch --gtest_filter='CommandQueueMultiDevice*Fixture.*'"
run_cmd "2.dispatch/UnitMeshCQSingleDevice" \
    "$WH_MULTICHIP_ENV TT_METAL_ENABLE_REMOTE_CHIP=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter='UnitMeshCQSingleDevice*Fixture.*'"
run_cmd "2.dispatch/UnitMeshCQMultiDevice" \
    "$WH_MULTICHIP_ENV ./build/test/tt_metal/unit_tests_dispatch --gtest_filter='UnitMeshCQMultiDevice*Fixture.*'"

run_cmd "2.debug_tools/mesh" \
    "$WH_MULTICHIP_ENV ./build/test/tt_metal/unit_tests_debug_tools --gtest_filter='DPrintMeshFixture.*:MeshWatcherFixture.*'"

for ex in distributed_program_dispatch distributed_buffer_rw distributed_eltwise_add distributed_trace_and_events; do
    run_cmd "2.examples/$ex" \
        "$WH_MULTICHIP_ENV ./build/programming_examples/distributed/$ex"
done

# TT-Fabric (ttsim: mock cluster + matching sim arch)
for filt in "ControlPlaneFixture.*T3k*" "T3kCustomMeshGraphControlPlaneTests*" "T3k*MeshGraphFabric2DDynamicTests*"; do
    run_cmd "2.fabric/control/$filt" \
        "$MOCK_FABRIC_ENV ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter='$filt'"
done

run_cmd "2.fabric/worker_edm" \
    "$WH_MULTICHIP_ENV ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter='*WorkerFabricEdmDatapath*:*EdmFabric*'"

run_cmd "2.fabric/unicast_1x8" \
    "$MOCK_FABRIC_ENV TT_MESH_GRAPH_DESC_PATH=tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_1x8_mesh_graph_descriptor.textproto ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter='*Fabric2DFixture.TestUnicast*'"

run_cmd "2.fabric/telemetry/Fabric2D" \
    "$WH_MULTICHIP_ENV TT_METAL_FABRIC_BW_TELEMETRY=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter='Fabric2D*Fixture.*'"
run_cmd "2.fabric/telemetry/Fabric1D" \
    "$WH_MULTICHIP_ENV TT_METAL_FABRIC_BW_TELEMETRY=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter='Fabric1D*Fixture.*'"
run_cmd "2.fabric/Fabric2D" \
    "$WH_MULTICHIP_ENV ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter='Fabric2D*Fixture.*'"
run_cmd "2.fabric/Fabric1D" \
    "$WH_MULTICHIP_ENV ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter='Fabric1D*Fixture.*'"
run_cmd "2.fabric/t3k_dynamic" \
    "$MOCK_FABRIC_ENV ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter='T3k*MeshGraphFabric2DDynamicTests*'"

for cfg in test_fabric_sanity_common.yaml test_fabric_sanity_at_least_2x2_mesh.yaml test_fabric_ubench_at_least_2x2_mesh.yaml; do
    run_cmd "2.fabric_ubench/$cfg" \
        "$WH_MULTICHIP_ENV ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config $TT_METAL_HOME/tests/tt_metal/tt_metal/perf_microbenchmark/routing/$cfg"
done

# TTNN distributed (T3K) — includes second unit_tests_ttnn from md
run_cmd "2.ttnn_dist/unit_tests_ttnn" "$WH_MULTICHIP_ENV ./build/test/ttnn/unit_tests_ttnn"
run_cmd "2.ttnn_dist/unit_tests_ttnn_udm" "$WH_MULTICHIP_ENV ./build/test/ttnn/unit_tests_ttnn_udm"
run_cmd "2.ttnn_dist/prefetcher" \
    "$WH_MULTICHIP_ENV $PYTHON -m pytest tests/ttnn/unit_tests/operations/transformers/test_prefetcher.py::test_run_prefetcher_post_commit_multi_device -xvvv --timeout-method=thread"
for py in test_tensor_parallel_example_T3000.py test_data_parallel_example.py test_hybrid_data_tensor_parallel_example_T3000.py; do
    run_cmd "2.ttnn_dist/$py" "$WH_MULTICHIP_ENV $PYTHON -m pytest tests/ttnn/distributed/$py -xvvv --timeout-method=thread"
done

# Multiprocess (tt-run) — not single-host; skip unless PARITY_INCLUDE_MP=1
if [ "$PARITY_INCLUDE_MP" = 1 ] && command -v tt-run >/dev/null 2>&1; then
    run_cmd "2.mp/2x2_fabric_ubench" \
        "tt-run --mpi-args '--allow-run-as-root --oversubscribe' --rank-binding tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_t3k_2x2.yaml"
    run_cmd "2.mp/multi_host_fabric" \
        "tt-run --mpi-args '--allow-run-as-root --oversubscribe' --rank-binding tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml ./build/test/tt_metal/multi_host_fabric_tests"
    run_cmd "2.mp/mesh_socket" \
        "tt-run --mpi-args '--allow-run-as-root --oversubscribe' --rank-binding tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml ./build/test/tt_metal/test_mesh_socket_main --test_config tests/tt_metal/multihost/fabric_tests/mesh_socket_t3k_2x2.yaml"
    run_cmd "2.mp/BigMeshDualRankTest2x4" \
        "tt-run --mpi-args '--allow-run-as-root --oversubscribe' --rank-binding tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml build/test/tt_metal/distributed/multiprocess/distributed_multiprocess_tests --gtest_filter='*BigMeshDualRankTest2x4*'"
    run_cmd "2.mp/BigMeshDualRankMeshShapeSweep" \
        "tt-run --mpi-args '--allow-run-as-root --oversubscribe' --rank-binding tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml build/test/tt_metal/distributed/multiprocess/distributed_multiprocess_tests --gtest_filter='*BigMeshDualRankMeshShapeSweep*'"
    run_cmd "2.mp/ttnn_dual_rank_2x2" \
        "tt-run --mpi-args '--allow-run-as-root --oversubscribe' --rank-binding tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml ./build/test/ttnn/multiprocess/unit_tests_dual_rank_2x2"
    run_cmd "2.mp/ttnn_dual_rank_2x4" \
        "tt-run --mpi-args '--allow-run-as-root --oversubscribe' --rank-binding tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml build/test/ttnn/multiprocess/unit_tests_dual_rank_2x4"
    run_cmd "2.mp/ttnn_launch_op" \
        "tt-run --mpi-args '--allow-run-as-root --oversubscribe' --rank-binding tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml build/test/ttnn/unit_tests_ttnn --gtest_filter='*LaunchOperation*'"
    run_cmd "2.mp/py_data_parallel" \
        "tt-run --mpi-args '--allow-run-as-root --oversubscribe' --rank-binding tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml pytest -svv tests/ttnn/distributed/test_data_parallel_example.py"
    run_cmd "2.mp/py_submesh" \
        "tt-run --mpi-args '--allow-run-as-root --oversubscribe' --rank-binding tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml pytest -svv tests/ttnn/distributed/test_submesh_not_spanning_all_ranks_T3000.py"
elif [ "$PARITY_INCLUDE_MP" = 1 ]; then
    log "SKIP multiprocess suite: tt-run not in PATH"
    echo -e "2.mp/all\tSKIP\t0\t0\ttt-run missing" >>"$SUMMARY"
fi

fi

if section_enabled 3; then
# --- Section 3: Galaxy ---
for py in test_data_parallel_example_TG.py test_multidevice_TG.py; do
    run_cmd "3.galaxy/$py" \
        "$WH_MULTICHIP_ENV $PYTHON -m pytest tests/ttnn/distributed/$py --timeout=900 -xvvv"
done
run_cmd "3.galaxy/multi_device_trace" \
    "$WH_MULTICHIP_ENV $PYTHON -m pytest tests/ttnn/unit_tests/base_functionality/test_multi_device_trace_TG.py --timeout=900 -xvvv"

fi

if section_enabled 4; then
# --- Section 4: LLK (craq-sim ttsim driver) ---
setup_llk_env
if [ -x "$CRAQ_SIM/scripts/llk-pytest-sweep.sh" ]; then
    run_cmd "4.llk/weekly_wh" \
        "$CRAQ_SIM/scripts/llk-pytest-sweep.sh weekly wh --timeout 120 --workers 2 --run-root $RESULTS_DIR/llk-weekly-wh"
    run_cmd "4.llk/nightly_wh" \
        "$CRAQ_SIM/scripts/llk-pytest-sweep.sh nightly wh --timeout 300 --workers 2 --run-root $RESULTS_DIR/llk-nightly-wh"
else
    LLK_PY="$TT_METAL_HOME/tt_metal/third_party/tt_llk/tests/.venv/bin/python"
    if [ -x "$LLK_PY" ]; then
        run_cmd "4.llk/weekly_inline" \
            "cd tt_metal/third_party/tt_llk/tests/python_tests && ../.venv/bin/python -m pytest -m 'not quasar and not nightly and not perf' --run-simulator --forked --timeout=120 -n 2 -q --maxfail=5 ."
        run_cmd "4.llk/nightly_inline" \
            "cd tt_metal/third_party/tt_llk/tests/python_tests && ../.venv/bin/python -m pytest -m nightly --run-simulator --forked --timeout=600 -n 4 test_unpack_matmul.py test_math_matmul.py test_zzz_eltwise_unary_sfpu.py"
    else
        echo -e "4.llk/all\tSKIP\t0\t0\tLLK venv missing" >>"$SUMMARY"
    fi
fi

fi

log "Sweep complete. Summary: $SUMMARY"
python3 - "$SUMMARY" <<'PY'
import sys
from collections import Counter
path = sys.argv[1]
counts = Counter()
with open(path) as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            counts[parts[1]] += 1
print("=== RESULT COUNTS ===")
for k in sorted(counts):
    print(f"  {k}: {counts[k]}")
PY

python3 "$REPO_ROOT/scripts/generate-parity-error-summary.py" "$RESULTS_DIR"
log "Error summary: $RESULTS_DIR/ERRORS.md"
log "Tail live output: tail -f $RUN_LOG"
