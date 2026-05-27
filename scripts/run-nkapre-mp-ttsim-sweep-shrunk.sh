#!/usr/bin/env bash
# Shrunk variant of run-nkapre-mp-ttsim-sweep.sh that
#   (1) substitutes minimal test configs / gtest filters under tt-sim,
#   (2) raises PER_CMD_TIMEOUT to 1800s (30 min) by default,
# so we can see which multi-process tests structurally complete (PASS)
# vs. still TIMEOUT under the simulator after the H_REUSED_DEV_PTR and
# H_CROSS_RANK_PEER_DROPPED fixes have landed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/require-bh-glx-compute.sh
source "${REPO_ROOT}/scripts/lib/require-bh-glx-compute.sh"
require_bh_glx_compute

BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build_Debug}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/craq-parity-results/mp-run-shrunk-$(date -u +%Y%m%dT%H%M%SZ)}"
PER_CMD_TIMEOUT="${PER_CMD_TIMEOUT:-1800}"

mkdir -p "$RESULTS_DIR"
SUMMARY="$RESULTS_DIR/summary.tsv"
RUN_LOG="$RESULTS_DIR/run.log"
: >"$SUMMARY"
: >"$RUN_LOG"
echo -e "suite\tstatus\texit_code\tduration_sec\tcommand" >>"$SUMMARY"

export TT_METAL_HOME="$REPO_ROOT"
export PYTHONPATH="${TT_METAL_HOME}${PYTHONPATH:+:$PYTHONPATH}"
export ARCH_NAME="${ARCH_NAME:-blackhole}"
# Mesh-Device gtest fixtures (multi_device_fixture.hpp:120) skip with
# GTEST_SKIP() whenever getenv("TT_METAL_SLOW_DISPATCH_MODE") returns a
# non-NULL string — they test "is set", not "value != 0". The unsetenv
# below lets the fast-dispatch path actually run under tt-sim.
unset TT_METAL_SLOW_DISPATCH_MODE
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
export ARCH_NAME=wormhole_b0
# tt-run blocks TT_METAL_MOCK_CLUSTER_DESC_PATH from parent env; per-rank mock
# descriptors must be passed via --mock-cluster-rank-binding (T3K big mesh).
MOCK_CLUSTER_RANK_BINDING="${MOCK_CLUSTER_RANK_BINDING:-tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_2x4_big_mesh_cluster_desc_mapping.yaml}"
TT_RUN_MOCK_ARGS="--mock-cluster-rank-binding ${MOCK_CLUSTER_RANK_BINDING}"

TTSIMD_BIN=""
TTSIM_RPC_SRC=""
for candidate in \
    "$CRAQ_SIM/daemon/_out/ttsimd" \
    "$CRAQ_SIM/src/_out/release_wh/ttsimd" \
    "$CRAQ_SIM/daemon/ttsimd"; do
    if [ -x "$candidate" ]; then
        TTSIMD_BIN="$candidate"
        break
    fi
done
for candidate in \
    "$CRAQ_SIM/client/_out/libttsim_rpc.so" \
    "$CRAQ_SIM/src/_out/release_wh/libttsim_rpc.so" \
    "$CRAQ_SIM/client/libttsim_rpc.so"; do
    if [ -f "$candidate" ]; then
        TTSIM_RPC_SRC="$candidate"
        break
    fi
done
if [ -n "$TTSIM_RPC_SRC" ]; then
    cp "$TTSIM_RPC_SRC" "$WH_SIM_DIR/libttsim_rpc.so"
fi

TTSIMD_PID=""
TTSIMD_SOCKET="${TTSIMD_SOCKET:-$RESULTS_DIR/ttsimd.sock}"

stop_ttsimd() {
    if [ -n "${TTSIMD_PID:-}" ] && kill -0 "$TTSIMD_PID" 2>/dev/null; then
        kill -TERM "$TTSIMD_PID" 2>/dev/null || true
        wait "$TTSIMD_PID" 2>/dev/null || true
    fi
}

start_ttsimd() {
    rm -f "$TTSIMD_SOCKET"
    "$TTSIMD_BIN" --socket "$TTSIMD_SOCKET" --libttsim "$WH_SIM_DIR/libttsim.so" \
        >>"$RESULTS_DIR/ttsimd.log" 2>&1 &
    TTSIMD_PID=$!

    local waited=0
    while [ "$waited" -lt 30 ]; do
        if [ -S "$TTSIMD_SOCKET" ]; then
            return 0
        fi
        if ! kill -0 "$TTSIMD_PID" 2>/dev/null; then
            echo "[mp-sweep-shrunk] ERROR: ttsimd exited early; see $RESULTS_DIR/ttsimd.log" | tee -a "$RUN_LOG"
            tail -20 "$RESULTS_DIR/ttsimd.log" >>"$RUN_LOG" 2>/dev/null || true
            exit 1
        fi
        sleep 0.1
        waited=$((waited + 1))
    done
    echo "[mp-sweep-shrunk] ERROR: ttsimd did not create socket within 3s" | tee -a "$RUN_LOG"
    exit 1
}

use_daemon=0
case "${TTSIM_USE_DAEMON:-auto}" in
    0|false|no|NO) use_daemon=0 ;;
    1|true|yes|YES)
        use_daemon=1
        if [ -z "$TTSIMD_BIN" ] || [ ! -f "$WH_SIM_DIR/libttsim_rpc.so" ]; then
            echo "[mp-sweep-shrunk] ERROR: TTSIM_USE_DAEMON=1 but ttsimd or libttsim_rpc.so not found under CRAQ_SIM=$CRAQ_SIM" | tee -a "$RUN_LOG"
            exit 1
        fi
        ;;
    auto|*)
        if [ -n "$TTSIMD_BIN" ] && [ -f "$WH_SIM_DIR/libttsim_rpc.so" ]; then
            use_daemon=1
        fi
        ;;
esac

if [ ! -e "$REPO_ROOT/build" ] || [ -L "$REPO_ROOT/build" ]; then
    ln -sfn "$(basename "$BUILD_DIR")" "$REPO_ROOT/build"
fi

log() { echo "[mp-sweep-shrunk] $*" | tee -a "$RUN_LOG"; }

if [ "$use_daemon" -eq 1 ]; then
    export TTSIMD_SOCKET
    export TT_TTSIMD_SOCKET="$TTSIMD_SOCKET"
    export TT_METAL_SIMULATOR="$WH_SIM_DIR/libttsim_rpc.so"
    # Client ranks use physical chip IDs; ttsimd is a separate process and must
    # match so eth peers and fabric handshakes target the same Device*.
    export TT_METAL_NO_CHIP_ID_REMAP=1
    export TTSIMD_PHYSICAL_CHIP_IDS=1
    # Each run_cmd starts its own ttsimd so a daemon crash in test N does not
    # poison test N+1. Trap still cleans up the last one on exit.
    trap stop_ttsimd EXIT INT TERM
    log "daemon mode (per-test ttsimd): TT_METAL_SIMULATOR=$TT_METAL_SIMULATOR TTSIMD_SOCKET=$TTSIMD_SOCKET mock=$MOCK_CLUSTER_RANK_BINDING timeout=${PER_CMD_TIMEOUT}s"
else
    log "in-process ttsim: TT_METAL_SIMULATOR=$TT_METAL_SIMULATOR timeout=${PER_CMD_TIMEOUT}s"
fi

run_cmd() {
    local suite="$1"; shift
    local cmd=("$@")
    local logfile="$RESULTS_DIR/$(echo "$suite" | tr '/ ' '__').log"
    local start end dur rc status

    # Allow targeting individual suites via SUITE_FILTER=substring (case-sensitive).
    if [ -n "${SUITE_FILTER:-}" ]; then
        case "$suite" in
            *"$SUITE_FILTER"*) ;;
            *) log "SKIP [$suite] (SUITE_FILTER=$SUITE_FILTER)"; return 0 ;;
        esac
    fi

    # In daemon mode each test gets a FRESH ttsimd. The daemon currently
    # dies (silently) part-way through some test workloads, which poisons
    # every subsequent test with `connect() to ttsimd failed`. Restarting
    # per-test isolates each suite so we can see real PASS/FAIL outcomes.
    if [ "$use_daemon" -eq 1 ]; then
        stop_ttsimd
        : >"$RESULTS_DIR/ttsimd.log"  # truncate so per-test daemon output is visible
        start_ttsimd
    fi

    log "RUN [$suite]: ${cmd[*]}"
    start=$(date +%s)
    set +e
    # ── Optional in-flight backtrace capture (BACKTRACE_AT_SEC=N) ────────
    # When set, after N seconds we attach gdb to all rank worker processes
    # spawned by tt-run/prterun and dump their stack traces to <suite>.gdb.log
    # before the timeout fires. Use when triaging silent hangs.
    if [ -n "${BACKTRACE_AT_SEC:-}" ]; then
        local gdb_log="$RESULTS_DIR/$(echo "$suite" | tr '/ ' '__').gdb.log"
        (
            sleep "$BACKTRACE_AT_SEC"
            {
                echo "==================================================="
                echo "[backtrace-watcher] $(date -u +%FT%TZ) capturing stacks for suite=$suite"
                # Worker patterns: test binary names appearing after `./build/...`
                # Pick everything that looks like a long-running test child.
                pgrep -af 'test_tt_fabric|test_mesh_socket_main|distributed_multiprocess_tests|unit_tests_dual_rank|unit_tests_ttnn|pytest' 2>/dev/null | grep -v 'pgrep' || true
                echo
                for pid in $(pgrep -f 'test_tt_fabric|test_mesh_socket_main|distributed_multiprocess_tests|unit_tests_dual_rank|unit_tests_ttnn|pytest' 2>/dev/null); do
                    echo "================ PID $pid ================"
                    cat "/proc/$pid/cmdline" 2>/dev/null | tr '\0' ' ' | head -c 400; echo
                    timeout 25 gdb -batch -p "$pid" -ex "set pagination off" -ex "thread apply all bt 25" 2>&1 | head -150
                    echo
                done
            } > "$gdb_log" 2>&1
        ) &
        local BT_WATCHER_PID=$!
    fi

    timeout "$PER_CMD_TIMEOUT" bash -lc "cd '$REPO_ROOT' && ${cmd[*]}" >"$logfile" 2>&1
    rc=$?

    if [ -n "${BACKTRACE_AT_SEC:-}" ] && [ -n "${BT_WATCHER_PID:-}" ]; then
        # Reap watcher if still running (test ended quickly)
        kill -TERM "$BT_WATCHER_PID" 2>/dev/null || true
        wait "$BT_WATCHER_PID" 2>/dev/null || true
    fi
    set -e
    end=$(date +%s); dur=$((end - start))

    # Snapshot per-suite daemon state for post-mortem.
    if [ "$use_daemon" -eq 1 ]; then
        cp "$RESULTS_DIR/ttsimd.log" "$RESULTS_DIR/$(echo "$suite" | tr '/ ' '__').ttsimd.log" 2>/dev/null || true
        if [ -n "${TTSIMD_PID:-}" ] && ! kill -0 "$TTSIMD_PID" 2>/dev/null; then
            log "  WARN [$suite] ttsimd (pid=$TTSIMD_PID) died during run"
        fi
    fi

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
    log "ERROR: tt-run not in PATH"; exit 127
fi

# 2.mp/2x2_fabric_ubench — single unicast, 1 packet, size 64.
run_cmd "2.mp/2x2_fabric_ubench_shrunk" \
    "tt-run --mpi-args '--allow-run-as-root --oversubscribe' ${TT_RUN_MOCK_ARGS} --rank-binding tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric --test_config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_t3k_2x2_ttsim.yaml"

# 2.mp/multi_host_fabric — smallest single gtest case.
run_cmd "2.mp/multi_host_fabric_shrunk" \
    "tt-run --mpi-args '--allow-run-as-root --oversubscribe' ${TT_RUN_MOCK_ARGS} --rank-binding tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml ./build/test/tt_metal/multi_host_fabric_tests --gtest_filter='InterMeshSplit1x2FabricFixture.MultiHopUnicast'"

# 2.mp/mesh_socket — single iteration, 1 transaction, 256-byte data.
run_cmd "2.mp/mesh_socket_shrunk" \
    "tt-run --mpi-args '--allow-run-as-root --oversubscribe' ${TT_RUN_MOCK_ARGS} --rank-binding tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml ./build/test/tt_metal/test_mesh_socket_main --test_config tests/tt_metal/multihost/fabric_tests/mesh_socket_t3k_2x2_ttsim.yaml"

# Distributed multiprocess tests (already small, just narrow gtest filter).
run_cmd "2.mp/BigMeshDualRankTest2x4" \
    "tt-run --mpi-args '--allow-run-as-root --oversubscribe' ${TT_RUN_MOCK_ARGS} --rank-binding tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml build/test/tt_metal/distributed/multiprocess/distributed_multiprocess_tests --gtest_filter='*BigMeshDualRankTest2x4*'"
run_cmd "2.mp/BigMeshDualRankMeshShapeSweep" \
    "tt-run --mpi-args '--allow-run-as-root --oversubscribe' ${TT_RUN_MOCK_ARGS} --rank-binding tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml build/test/tt_metal/distributed/multiprocess/distributed_multiprocess_tests --gtest_filter='*BigMeshDualRankMeshShapeSweep*'"

# ttnn dual-rank send/recv: just the first parametrized case per fixture.
run_cmd "2.mp/ttnn_dual_rank_2x2_shrunk" \
    "tt-run --mpi-args '--allow-run-as-root --oversubscribe' ${TT_RUN_MOCK_ARGS} --rank-binding tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml ./build/test/ttnn/multiprocess/unit_tests_dual_rank_2x2 --gtest_filter='MeshDeviceSplit2x2SendRecvTests/MeshDeviceSplit2x2SendRecvFixture.SendRecvAsync/0'"
run_cmd "2.mp/ttnn_dual_rank_2x4_shrunk" \
    "tt-run --mpi-args '--allow-run-as-root --oversubscribe' ${TT_RUN_MOCK_ARGS} --rank-binding tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml build/test/ttnn/multiprocess/unit_tests_dual_rank_2x4 --gtest_filter='BigMeshDualRankTest2x4.HostAllGather'"

run_cmd "2.mp/ttnn_launch_op" \
    "tt-run --mpi-args '--allow-run-as-root --oversubscribe' ${TT_RUN_MOCK_ARGS} --rank-binding tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml build/test/ttnn/unit_tests_ttnn --gtest_filter='*LaunchOperation*'"

run_cmd "2.mp/py_data_parallel" \
    "tt-run --mpi-args '--allow-run-as-root --oversubscribe' ${TT_RUN_MOCK_ARGS} --rank-binding tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml pytest -svv tests/ttnn/distributed/test_data_parallel_example.py"
run_cmd "2.mp/py_submesh" \
    "tt-run --mpi-args '--allow-run-as-root --oversubscribe' ${TT_RUN_MOCK_ARGS} --rank-binding tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml pytest -svv tests/ttnn/distributed/test_submesh_not_spanning_all_ranks_T3000.py"

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
print("=== MP SHRUNK RESULT COUNTS ===")
for k in sorted(counts):
    print(f"  {k}: {counts[k]}")
PY
