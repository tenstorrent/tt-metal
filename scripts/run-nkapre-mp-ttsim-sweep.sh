#!/usr/bin/env bash
# Run only section 2.mp tt-run multiprocess suites from nkapre-fork-test-commands.md
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/require-bh-glx-compute.sh
source "${REPO_ROOT}/scripts/lib/require-bh-glx-compute.sh"
require_bh_glx_compute

BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build_Debug}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/craq-parity-results/mp-run-$(date -u +%Y%m%dT%H%M%SZ)}"
PER_CMD_TIMEOUT="${PER_CMD_TIMEOUT:-900}"

mkdir -p "$RESULTS_DIR"
SUMMARY="$RESULTS_DIR/summary.tsv"
RUN_LOG="$RESULTS_DIR/run.log"
: >"$SUMMARY"
: >"$RUN_LOG"
echo -e "suite\tstatus\texit_code\tduration_sec\tcommand" >>"$SUMMARY"

export TT_METAL_HOME="$REPO_ROOT"
export PYTHONPATH="${TT_METAL_HOME}${PYTHONPATH:+:$PYTHONPATH}"
export ARCH_NAME="${ARCH_NAME:-blackhole}"
# Fast dispatch required by MeshDeviceFixture-based MP suites; fabric ubench still runs under sim.
export TT_METAL_SLOW_DISPATCH_MODE="${TT_METAL_SLOW_DISPATCH_MODE:-0}"
export TT_METAL_DISABLE_SFPLOADMACRO=1
export TT_METAL_DRAM_BACKED_CQ=1
export TT_METAL_SIMULATOR_CQ_WAIT_CLOCKS=10000
export LD_LIBRARY_PATH="${BUILD_DIR}/tt_metal:${BUILD_DIR}/tt_metal/third_party/umd/lib:${BUILD_DIR}/ttnn:${BUILD_DIR}/tt_stl:${BUILD_DIR}/lib:${LD_LIBRARY_PATH:-}"
export PATH="${REPO_ROOT}/python_env/bin:${PATH}"

CRAQ_SIM="${CRAQ_SIM:-/data/rsong/craq-sim}"
WH_SIM_DIR="$RESULTS_DIR/sim_wh_multichip"
mkdir -p "$WH_SIM_DIR"
cp "$CRAQ_SIM/src/_out/release_wh/libttsim.so" "$WH_SIM_DIR/libttsim.so"
cp "$TT_METAL_HOME/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml" "$WH_SIM_DIR/soc_descriptor.yaml"
export TT_METAL_SIMULATOR="$WH_SIM_DIR/libttsim.so"
export TT_METAL_SIMULATOR_HOME="$WH_SIM_DIR"
export TT_METAL_MOCK_CLUSTER_DESC_PATH="${TT_METAL_MOCK_CLUSTER_DESC_PATH:-tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml}"
export ARCH_NAME=wormhole_b0

if [ ! -e "$REPO_ROOT/build" ] || [ -L "$REPO_ROOT/build" ]; then
    ln -sfn "$(basename "$BUILD_DIR")" "$REPO_ROOT/build"
fi

log() {
    echo "[mp-sweep] $*" | tee -a "$RUN_LOG"
}

run_cmd() {
    local suite="$1"
    shift
    local cmd=("$@")
    local logfile="$RESULTS_DIR/$(echo "$suite" | tr '/ ' '__').log"
    local start end dur rc status

    log "RUN [$suite]: ${cmd[*]}"
    start=$(date +%s)
    set +e
    timeout "$PER_CMD_TIMEOUT" bash -lc "cd '$REPO_ROOT' && ${cmd[*]}" >"$logfile" 2>&1
    rc=$?
    set -e
    end=$(date +%s)
    dur=$((end - start))

    case "$rc" in
        0) status=PASS ;;
        124) status=TIMEOUT ;;
        127) status=MISSING ;;
        *) status=FAIL ;;
    esac

    log "DONE [$suite] status=$status rc=$rc duration=${dur}s"
    echo -e "${suite}\t${status}\t${rc}\t${dur}\t${cmd[*]}" >>"$SUMMARY"
    tail -20 "$logfile" >>"$RUN_LOG" || true
}

if ! command -v tt-run >/dev/null 2>&1; then
    log "ERROR: tt-run not in PATH"
    exit 127
fi

if [ "${MP_SKIP_ON_SIM:-0}" = "1" ] && [ -n "${TT_METAL_SIMULATOR:-}" ]; then
    log "SKIP all mp suites: tt-run + ttsim unsupported"
    for mp_suite in \
        2.mp/2x2_fabric_ubench \
        2.mp/multi_host_fabric \
        2.mp/mesh_socket \
        2.mp/BigMeshDualRankTest2x4 \
        2.mp/BigMeshDualRankMeshShapeSweep \
        2.mp/ttnn_dual_rank_2x2 \
        2.mp/ttnn_dual_rank_2x4 \
        2.mp/ttnn_launch_op \
        2.mp/py_data_parallel \
        2.mp/py_submesh; do
        echo -e "${mp_suite}\tSKIP\t0\t0\tttsim skip: tt-run" >>"$SUMMARY"
    done
    log "Summary: $SUMMARY"
    exit 0
fi

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

log "Summary: $SUMMARY"
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
print("=== MP RESULT COUNTS ===")
for k in sorted(counts):
    print(f"  {k}: {counts[k]}")
PY
