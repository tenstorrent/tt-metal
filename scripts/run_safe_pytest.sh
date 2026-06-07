#!/bin/bash
# run_safe_pytest.sh - Cooperative device-aware test runner
#
# Uses flock to serialize device access across multiple agents/terminals.
# Uses TT_METAL_OPERATION_TIMEOUT_SECONDS for precise hang detection at the
# dispatch layer (does not penalize setup/compilation time).
# Automatically resets the device after hangs, ensuring the next runner
# always gets a clean device.
#
# Simulator mode (TT_METAL_SIMULATOR set):
#   Exports TT_METAL_SLOW_DISPATCH_MODE=1 and TT_METAL_DISABLE_SFPLOADMACRO=1.
#   Skips flock, device resets, and triage (these require real hardware).
#   No hang protection — sim runs at kHz, so wall-clock timeouts are meaningless.
#
# Usage: scripts/run_safe_pytest.sh [--dev] [--run-all] [--precompile] <test_path> [extra_pytest_args...]
#
# Options:
#   --dev       Enables polling watcher (NoC sanitizer, waypoints, CB
#               sanitization), lightweight ebreak asserts, and auto-triage
#               on hang with full triage + watcher log dump.
#   --run-all   Run all tests instead of stopping on first failure (-x).
#               Useful for eval scoring where you need full pass/fail counts.
#   --precompile  Before the real run, transparently warm the JIT cache hardware-free
#               and in parallel (no device, no env vars, no second command), so kernels
#               compile up-front in parallel instead of inline & serial. ALWAYS falls
#               back to a normal cold run if anything goes wrong — it can only make a
#               run slower, never broken or wrong. Prints a one-line diagnostic saying
#               whether the warm cache was hit, and if not, exactly why. Hardware only.
#               Tune parallelism with --precompile-workers N (default: nproc).
#
# Modes:
#   default  - Dispatch timeout only. Lean, no debug overhead.
#   --dev    - Debug mode with watcher, asserts, and triage (see above).
#
# Exit codes:
#   0 - All tests passed
#   1 - Test failure (normal pytest failure, no hang)
#   2 - Hang detected (dispatch timeout fired)
#   3 - Setup error (missing args, etc.)
#
# Total runtime:
#   Always prints SAFE_PYTEST_TOTAL_RUNTIME as the very last line (on every exit path).
#   It is the wall-clock time from "device lock acquired" (idle lock-wait queueing is
#   deliberately excluded) to script exit, so it covers the whole run — device reset,
#   precompile warm phase, and the pytest run itself — not just pytest. Under simulator
#   there is no lock, so the clock starts at the equivalent point.

set -o pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DISPATCH_TIMEOUT="${SAFE_PYTEST_DISPATCH_TIMEOUT:-5}"
TRIAGE_SCRIPT="${REPO_DIR}/tools/tt-triage.py"
WATCHER_LOG="${REPO_DIR}/generated/watcher/watcher.log"
LOCK_FILE="/tmp/tt-device.lock"
DIRTY_FLAG="/tmp/tt-device.dirty"
TRIAGE_LOG="/tmp/safe-pytest-triage-$$.log"

# --- Detect simulator mode ---
SIM_MODE=false
if [[ -n "${TT_METAL_SIMULATOR:-}" ]]; then
    SIM_MODE=true
    export TT_METAL_SLOW_DISPATCH_MODE=1
    export TT_METAL_DISABLE_SFPLOADMACRO=1
fi

# --- Parse flags ---
DEV_MODE=false
FAIL_FAST=true
PRECOMPILE=false
PRECOMPILE_WORKERS="${PRECOMPILE_WORKERS:-$(nproc 2>/dev/null || echo 8)}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev)
            DEV_MODE=true
            shift
            ;;
        --run-all)
            FAIL_FAST=false
            shift
            ;;
        --precompile)
            # Opt-in: transparently warm the JIT cache (hardware-free, parallel) before the
            # real run, so kernels compile up-front in parallel instead of inline & serial.
            # Everything is internal — no env vars, no second command. Always falls back to a
            # normal cold run if anything goes wrong (see precompile_warm). Hardware only.
            PRECOMPILE=true
            shift
            ;;
        --precompile-workers)
            PRECOMPILE_WORKERS="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# --- Argument validation ---
if [[ $# -eq 0 ]]; then
    echo "SAFE_PYTEST_ERROR: No test path provided" >&2
    echo "Usage: scripts/run_safe_pytest.sh [--dev] [--run-all] <test_path> [extra_pytest_args...]" >&2
    exit 3
fi

TEST_PATH="$1"
shift
# Remaining args are extra pytest args — the precompile collect must use the SAME selection.
EXTRA_ARGS=("$@")

# Precompile uses WHATEVER cache the user already has (TT_METAL_CACHE if set, else tt-metal's
# default) — both the warm-collect and the real run inherit the same value, so they share it and a
# pre-warmed cache is transparently reused. We never override it. The cluster descriptor (HW-stable)
# is cached in /tmp; the build_key is recomputed each run (it depends on source, not just HW).
PRECOMPILE_DESC="/tmp/tt_precompile_cluster_desc.yaml"
# Build fingerprint: the real device's build-determining values that a hardware-free (slow-dispatch)
# build resolves differently (num_l1_banks, dispatch core type/axis, resolved 2-erisc). Captured fresh
# each run (source-dependent, like build_key) and replayed in the mock via TT_METAL_JIT_BUILD_FINGERPRINT.
PRECOMPILE_FINGERPRINT="/tmp/tt_precompile_build_fingerprint.txt"
PRECOMPILE_PLUGIN_DIR="$REPO_DIR"

# ============================================================================
# Precompile (opt-in --precompile): warm the JIT cache hardware-free & parallel,
# then let the normal run below hit it. A definitive build_key PRE-FLIGHT decides
# whether warming can help BEFORE doing any work; every failure path degrades to
# a normal cold run — it can make a run slower, never broken or wrong.
# ============================================================================
_precompile_descriptor() {
    # Cluster descriptor from UMD topology (HW-stable -> cache it per container). 0 ok / 1 fail.
    [[ -f "$PRECOMPILE_DESC" ]] && return 0
    timeout 120 python3 - "$PRECOMPILE_DESC" >"/tmp/precompile_desc_$$.log" 2>&1 <<'PY'
import sys, tt_umd
tt_umd.TopologyDiscovery.create_cluster_descriptor().serialize_to_file(sys.argv[1])
PY
}

_precompile_realkey() {
    # Brief REAL device open -> prints "RKEY <2erisc> <build_key>" AND writes the build fingerprint
    # (num_l1_banks, dispatch core type/axis, resolved 2-erisc) to $PRECOMPILE_FINGERPRINT. build_key
    # depends on source, so this is recomputed every run (not cached). MUST open the device the same
    # way the tests do (open_mesh_device) — the build_key differs between single-device and mesh paths.
    timeout 180 env PRECOMPILE_FP="$PRECOMPILE_FINGERPRINT" PYTHONPATH="$REPO_DIR" \
        python3 - >"/tmp/precompile_real_$$.log" 2>&1 <<'PY'
import os, ttnn
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
try:
    f = 1 if ttnn.cluster.get_enable_2_erisc_mode() else 0
    k = ttnn.cluster.get_build_key()
    ttnn.cluster.capture_jit_build_fingerprint(os.environ["PRECOMPILE_FP"])
finally:
    ttnn.close_mesh_device(md)
print(f"RKEY {f} {k}")
PY
    grep '^RKEY ' "/tmp/precompile_real_$$.log" 2>/dev/null | tail -1
}

_precompile_mockkey() {
    # $1=2erisc. Hardware-free mock open replaying the captured build fingerprint -> "MKEY <key>".
    timeout 120 env \
        TT_METAL_FORCE_2_ERISC_MODE="$1" TT_METAL_JIT_BUILD_FINGERPRINT="$PRECOMPILE_FINGERPRINT" \
        TT_METAL_SLOW_DISPATCH_MODE=1 \
        TT_METAL_MOCK_CLUSTER_DESC_PATH="$PRECOMPILE_DESC" PYTHONPATH="$REPO_DIR" \
        python3 - >"/tmp/precompile_mock_$$.log" 2>&1 <<'PY'
import ttnn
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
try:
    print("MKEY", ttnn.cluster.get_build_key())
finally:
    ttnn.close_mesh_device(md)
PY
    grep '^MKEY ' "/tmp/precompile_mock_$$.log" 2>/dev/null | tail -1
}

precompile_warm() {
    [[ "$SIM_MODE" == false ]] && touch "$DIRTY_FLAG"
    echo "PRECOMPILE: ===== warming JIT cache (hardware-free) =====" >&2

    # 1. cluster descriptor (HW-stable, cached per container)
    if ! _precompile_descriptor; then
        echo "PRECOMPILE: ✗ cluster-descriptor capture failed/timed out -> COLD (real run UNAFFECTED). See /tmp/precompile_desc_$$.log" >&2
        return 0
    fi

    # 2. real device fingerprint: resolved 2-erisc + the build_key your run will use, and the build
    #    fingerprint file (num_l1_banks, dispatch core type/axis) written to $PRECOMPILE_FINGERPRINT
    local probe force2 realkey
    probe="$(_precompile_realkey | sed -n 's/^RKEY //p' | tail -1)"
    if [[ -z "$probe" ]]; then
        echo "PRECOMPILE: ✗ couldn't read your device's build_key -> COLD. Either the device is unhealthy, or" >&2
        echo "PRECOMPILE:   the build predates the get_build_key/capture_jit_build_fingerprint bindings (re-run ./build_metal.sh)." >&2
        echo "PRECOMPILE:   See /tmp/precompile_real_$$.log" >&2
        return 0
    fi
    read -r force2 realkey <<< "$probe"

    # 3. mock build_key (hardware-free), replaying the captured build fingerprint
    local mockkey
    mockkey="$(_precompile_mockkey "$force2" | sed -n 's/^MKEY //p' | tail -1)"
    if [[ -z "$mockkey" ]]; then
        echo "PRECOMPILE: ✗ couldn't compute the hardware-free build_key -> COLD. See /tmp/precompile_mock_$$.log" >&2
        return 0
    fi

    # 4. PRE-FLIGHT: will a warm pass actually be reused by your run?
    if [[ "$mockkey" != "$realkey" ]]; then
        echo "PRECOMPILE: ✗ build_key MISMATCH — your device uses ${realkey}, the hardware-free fingerprint produces ${mockkey}." >&2
        echo "PRECOMPILE:   => a warm pass would NOT be reused by your run, so it is SKIPPED (no wasted work); running COLD." >&2
        echo "PRECOMPILE:   Results stay CORRECT — you just don't get the speedup. Cause (mock fingerprint != real device):" >&2
        echo "PRECOMPILE:     • stale descriptor from another machine/docker -> rm -f $PRECOMPILE_DESC ; re-run" >&2
        echo "PRECOMPILE:     • a multi-device / Blackhole config the (1,1) hardware-free fingerprint didn't reproduce" >&2
        echo "PRECOMPILE:       (harvesting / dispatch_core / 2-erisc / num_l1_banks / arch)." >&2
        return 0
    fi
    echo "PRECOMPILE: ✓ fingerprint matches your device (build_key ${realkey}) — the warm cache WILL be reused." >&2

    # 5. hardware-free meta-collect over the SAME selection -> warms the shared cache.
    #    SINGLE-PROCESS by design: the heavy kernel COMPILE is parallelized by the plugin's in-process
    #    C++ thread pool via ttnn.graph.up_front_compile(device, UP_FRONT_COLLECT_WORKERS=N, ...). xdist
    #    (-n) would only parallelize the cheap COLLECT body-run across processes — and, measured, an
    #    xdist multi-process prewarm LOSES ~half the cache (concurrent writers to the shared on-disk
    #    cache + per-worker dedup): full conv2d warm hit 47.6% via xdist vs 99.8% single-process, same
    #    fix/build. So we always run one process with N compile-threads. A non-zero collect is surfaced
    #    (below), not swallowed by `|| true`.
    local nflag=() collect_workers="$PRECOMPILE_WORKERS"
    echo "PRECOMPILE: warming (single proc x ${PRECOMPILE_WORKERS} compile-threads) over: ${TEST_PATH} ${EXTRA_ARGS[*]}" >&2
    local clog="/tmp/precompile_collect_$$.log" t0 t1 cstatus
    t0=$(date +%s)
    TT_METAL_FORCE_2_ERISC_MODE="$force2" TT_METAL_JIT_BUILD_FINGERPRINT="$PRECOMPILE_FINGERPRINT" \
    TT_METAL_SLOW_DISPATCH_MODE=1 \
    TT_METAL_MOCK_CLUSTER_DESC_PATH="$PRECOMPILE_DESC" \
    UP_FRONT_COLLECT=1 UP_FRONT_META_COLLECT=1 UP_FRONT_COLLECT_WORKERS="$collect_workers" \
    LOGURU_LEVEL=ERROR PYTHONPATH="$PRECOMPILE_PLUGIN_DIR" \
        pytest "${TEST_PATH}" "${EXTRA_ARGS[@]}" -p up_front_collect_plugin "${nflag[@]}" \
        > "$clog" 2>&1
    cstatus=$?
    t1=$(date +%s)
    # Don't pretend it warmed if the collect failed. A non-zero exit (pytest usage/collection error,
    # plugin failure, etc.) means we warmed nothing -> say so plainly; the real run still runs COLD and
    # CORRECT, just without the speedup. (pytest exit 5 = "no tests collected" counts as a failure here.)
    if [[ $cstatus -ne 0 ]]; then
        echo "PRECOMPILE: ✗ warm collect FAILED (pytest exit $cstatus) after $((t1-t0))s -> warmed NOTHING; running COLD." >&2
        grep -iE "error|unrecognized|no tests ran|no tests collected" "$clog" 2>/dev/null | head -3 | sed 's/^/PRECOMPILE:   /' >&2
        echo "PRECOMPILE:   (full collect log: $clog)" >&2
        return 0
    fi
    echo "PRECOMPILE: ✓ warm pass complete in $((t1-t0))s (build_key ${realkey}) — the real run below reuses it. Log: $clog" >&2
}

# --- Total-run timer ---
# Reports wall-clock time from "device lock acquired" to script exit. Registered as an
# EXIT trap so it ALWAYS prints last, on every exit path (pass, fail, hang, error). The
# RUN_START guard means nothing is printed for exits that happen before testing begins
# (e.g. missing args), since no run took place.
_print_total_runtime() {
    [[ -z "${RUN_START:-}" ]] && return 0
    local run_end elapsed
    run_end=$(date +%s)
    elapsed=$((run_end - RUN_START))
    echo "========================================" >&2
    printf 'SAFE_PYTEST_TOTAL_RUNTIME: %dm%02ds (%ds total, device-lock-acquired -> exit)\n' \
        $((elapsed / 60)) $((elapsed % 60)) "$elapsed" >&2
}
trap _print_total_runtime EXIT

# --- Acquire flock (hardware only) ---
if [[ "$SIM_MODE" == false ]]; then
    exec 9>"$LOCK_FILE"

    echo "SAFE_PYTEST: Waiting for device lock..." >&2
    flock 9
    echo "SAFE_PYTEST: Device lock acquired" >&2

    # Start the total-run clock the moment we own the device. The lock-wait above is
    # idle queueing behind other runners and is deliberately excluded.
    RUN_START=$(date +%s)

    # --- Check if device needs reset from previous hang ---
    if [[ -f "$DIRTY_FLAG" ]]; then
        echo "SAFE_PYTEST: Device marked dirty from previous hang, resetting..." >&2
        if ! tt-smi -r; then
            echo "SAFE_PYTEST_ERROR: Device reset (tt-smi -r) failed" >&2
            exit 3
        fi
        rm -f "$DIRTY_FLAG"
        echo "SAFE_PYTEST: Device reset complete" >&2
    fi
else
    # Simulator: no device lock to acquire, so start the total-run clock here (the
    # equivalent "start of testing" point).
    RUN_START=$(date +%s)
fi

# --- Setup environment ---
cd "$REPO_DIR"
if [[ -f python_env/bin/activate ]]; then
    if ! source python_env/bin/activate; then
        echo "SAFE_PYTEST: WARNING: Failed to activate python_env virtual environment" >&2
    fi
else
    echo "SAFE_PYTEST: WARNING: python_env not found; using system Python" >&2
fi

# --- Precompile warm phase (opt-in, hardware only; never aborts the real run) ---
if [[ "$PRECOMPILE" == true ]]; then
    if [[ "$SIM_MODE" == true ]]; then
        echo "PRECOMPILE: skipped under simulator (no warm benefit)" >&2
    else
        precompile_warm
    fi
fi

# --- Hang detection setup (hardware only) ---
# On timeout, the dispatch layer runs tt-triage. Fires only on actual hang —
# zero overhead for passing tests. On sim there is no hang detection because
# wall-clock timeouts are meaningless at kHz clock speeds.
rm -f "$TRIAGE_LOG"
if [[ "$SIM_MODE" == false ]]; then
    export TT_METAL_OPERATION_TIMEOUT_SECONDS="$DISPATCH_TIMEOUT"
    # Requires tt-exalens: uv pip install -r tools/triage/requirements.txt
    if ! python3 -c "import ttexalens" 2>/dev/null; then
        echo "SAFE_PYTEST: WARNING: tt-exalens not installed — triage on hang will be unavailable." >&2
        echo "SAFE_PYTEST: Install with: uv pip install -r tools/triage/requirements.txt" >&2
    fi
    export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="python3 ${TRIAGE_SCRIPT} --disable-progress > ${TRIAGE_LOG} 2>&1"
fi

if [[ "$DEV_MODE" == true ]]; then
    # Lightweight asserts: compiles ASSERT() as ebreak, halting the core at the
    # exact instruction. The dispatch timeout then fires and runs triage, which
    # captures callstacks from ALL cores — showing both the assert site and any
    # cores blocked waiting on the halted one.
    export TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1

    # LLK asserts: enables LLK_ASSERT() in the compute API / LLK layer.
    # Catches invalid hardware configurations, wrong unpack/pack parameters,
    # and API misuse deep in the compute pipeline. Also compiles as ebreak.
    export TT_METAL_LLK_ASSERTS=1

    # Polling watcher: enables all device-side instrumentation (NoC sanitizer,
    # waypoints, CB sanitization) with the host polling thread. Recent watcher
    # server improvements (t=0 sampling, 100ms quanta) minimize overhead for
    # short tests. Watcher log is dumped on hang for full diagnostic context.
    #
    # WATCHER_DISABLE_ASSERT disables the watcher's own assert mechanism (which
    # would conflict with the lightweight ebreak asserts above). We want ebreak
    # asserts to halt the core so triage can capture callstacks from all cores,
    # rather than having watcher handle asserts independently.
    export TT_METAL_WATCHER=1
    export TT_METAL_WATCHER_NOINLINE=1
    export TT_METAL_WATCHER_DISABLE_ASSERT=1
    export TT_METAL_WATCHER_DISABLE_DISPATCH=1

    if [[ "$SIM_MODE" == true ]]; then
        echo "SAFE_PYTEST: [sim+dev] asserts=ebreak llk_asserts=ON watcher=polling (no hang detection on sim)" >&2
    else
        echo "SAFE_PYTEST: [dev] asserts=ebreak llk_asserts=ON watcher=polling triage=ON timeout=${DISPATCH_TIMEOUT}s" >&2
    fi
elif [[ "$SIM_MODE" == true ]]; then
    echo "SAFE_PYTEST: [sim] no hang detection" >&2
else
    echo "SAFE_PYTEST: dispatch_timeout=${DISPATCH_TIMEOUT}s" >&2
fi
echo "SAFE_PYTEST: pytest ${TEST_PATH} $*" >&2
echo "========================================" >&2

# --- Mark device dirty before running tests (hardware only) ---
# Pessimistic: assume the device will get corrupted. If the script is killed at any
# point (SIGKILL, OOM, etc.), the flag persists and the next runner will reset.
# Cleared on clean exit or after a successful inline reset.
if [[ "$SIM_MODE" == false ]]; then
    touch "$DIRTY_FLAG"
fi

# --- Run pytest ---
# -x: stop on first failure (avoids running tests after a hang bricks the device)
# --run-all: skip -x to get full pass/fail counts (for eval scoring)
PYTEST_CMD=(pytest "${TEST_PATH}")
if [[ "$FAIL_FAST" == true ]]; then
    PYTEST_CMD+=(-x)
fi
PYTEST_CMD+=("$@")

"${PYTEST_CMD[@]}"
EXIT_CODE=$?

echo "========================================" >&2

# --- Handle result ---
if [[ $EXIT_CODE -eq 0 ]]; then
    rm -f "$DIRTY_FLAG"
    rm -f "$TRIAGE_LOG"
    echo "SAFE_PYTEST_RESULT: PASS" >&2
    exit 0
fi

# Pytest setup errors (no device touched):
#   4 = usage error (bad args, nonexistent path)
#   5 = no tests collected (typo in path, bad marker filter, etc.)
if [[ $EXIT_CODE -eq 4 || $EXIT_CODE -eq 5 ]]; then
    rm -f "$DIRTY_FLAG"
    rm -f "$TRIAGE_LOG"
    if [[ $EXIT_CODE -eq 4 ]]; then
        echo "SAFE_PYTEST_ERROR: Pytest usage error (invalid path or arguments)" >&2
    else
        echo "SAFE_PYTEST_ERROR: No tests collected" >&2
    fi
    exit 3
fi

# Kill any remaining child processes (pytest may have left orphans)
pkill -9 -P $$ 2>/dev/null || true

# Determine if this was a hang:
#   Triage log non-empty = dispatch timeout handler ran tt-triage (definitive hang signal)
# On sim there is no hang detection, so IS_HANG always stays false.
IS_HANG=false
if [[ -s "$TRIAGE_LOG" ]]; then
    IS_HANG=true
fi

# Only reset device when the failure might have left it dirty.
# Hangs and crashes corrupt device state. Normal test failures (PCC mismatch,
# assertion errors) and collection errors don't touch the device.
if [[ "$IS_HANG" == true ]]; then
    echo "SAFE_PYTEST: Resetting device..." >&2
    if tt-smi -r; then
        sleep 2
        rm -f "$DIRTY_FLAG"
        echo "SAFE_PYTEST: Device reset complete" >&2
    else
        echo "SAFE_PYTEST: Device reset FAILED; leaving device marked dirty" >&2
    fi

    echo "SAFE_PYTEST_RESULT: HANG (exit code: $EXIT_CODE)" >&2
    echo "" >&2

    # Dump full triage log
    echo "=== TRIAGE LOG ===" >&2
    cat "$TRIAGE_LOG" >&2
    echo "=== END TRIAGE LOG ===" >&2
    echo "" >&2

    # In dev mode, also dump watcher log
    if [[ "$DEV_MODE" == true && -f "$WATCHER_LOG" ]]; then
        echo "=== WATCHER LOG (last 50 lines) ===" >&2
        tail -50 "$WATCHER_LOG" >&2
        echo "=== END WATCHER LOG ===" >&2
        echo "" >&2
    fi

    rm -f "$TRIAGE_LOG"
    exit 2
fi

rm -f "$DIRTY_FLAG"
rm -f "$TRIAGE_LOG"
echo "SAFE_PYTEST_RESULT: FAIL (exit code: $EXIT_CODE)" >&2
exit 1
