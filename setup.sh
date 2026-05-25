# =============================================================================
# setup.sh — emule + ASAN environment for ttnn test runs
#
#   USAGE:  source setup.sh        (NOT ./setup.sh — env vars must persist)
#
# After sourcing, run a hardened runner (./eltwise_tests.sh, ./data_mov_tests.sh,
# ./base_func_tests.sh). Each one calls emule_preflight before and
# emule_postflight after, so a misconfigured run can't masquerade as a clean one.
# =============================================================================

# Refuse to be executed instead of sourced (exports would be lost).
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "ERROR: source this file, do not execute it:  source setup.sh"
    exit 1
fi

export TT_METAL_HOME="$(pwd)"
export TT_METAL_RUNTIME_ROOT="$(pwd)"
export PYTHONPATH="$(pwd)"
export ARCH_NAME=wormhole_b0

# --- Route to the tt-emule software emulator (NOT the physical WH card) -------
# Without these, tt-metal defaults to TargetDevice::Hardware and opens the real
# device — emulated_program_runner.cpp and every ASAN sanitizer are bypassed.
export TT_METAL_EMULE_MODE=1
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_MOCK_CLUSTER_DESC_PATH="$TT_METAL_HOME/tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml"

# --- Arm the sanitizers -------------------------------------------------------
export TT_METAL_EMULE_ASAN=1

unset PYTHON_ENV_DIR
source "$(pwd)/python_env/bin/activate"

# --- Guards shared by the *_tests.sh runners ---------------------------------
# Verifies the config BEFORE a run. Returns non-zero (does not exit) so it is
# safe to call while sourced.
emule_preflight() {
    local ok=1
    [ "${TT_METAL_EMULE_MODE:-}" = "1" ] || { echo "[preflight] FATAL: TT_METAL_EMULE_MODE != 1"; ok=0; }
    [ "${TT_METAL_EMULE_ASAN:-}" = "1" ] || { echo "[preflight] FATAL: TT_METAL_EMULE_ASAN != 1"; ok=0; }
    [ "${TT_METAL_SLOW_DISPATCH_MODE:-}" = "1" ] || { echo "[preflight] FATAL: TT_METAL_SLOW_DISPATCH_MODE != 1"; ok=0; }
    [ -f "${TT_METAL_MOCK_CLUSTER_DESC_PATH:-/nonexistent}" ] || { echo "[preflight] FATAL: mock cluster desc not found: ${TT_METAL_MOCK_CLUSTER_DESC_PATH:-unset}"; ok=0; }
    # Use grep -c into a var (not `grep -q`): grep -q closes the pipe early, which
    # gives nm SIGPIPE and, under `set -o pipefail`, falsely fails the pipeline.
    local sym_count
    sym_count=$(nm -DC "$TT_METAL_HOME/build_emule/tt_metal/libtt_metal.so" 2>/dev/null | grep -c "emule::execute_program_emulated") || true
    if [ "${sym_count:-0}" -eq 0 ]; then
        echo "[preflight] FATAL: libtt_metal.so is not an EMULE build (missing emule::execute_program_emulated)"; ok=0
    fi
    [ "$ok" = "1" ] || { echo "[preflight] environment NOT ready — fix the above and re-source setup.sh"; return 1; }
    echo "[preflight] OK: emule build + emule mode + slow dispatch + ASAN all set."
    return 0
}

# Verifies the emulator actually ran AFTER a run, then reports ASAN hits.
# Usage: emule_postflight <logfile>
emule_postflight() {
    local log="$1"
    if ! grep -q "execute_program_emulated" "$log"; then
        echo "[postflight] INVALID RUN: 'execute_program_emulated' never appeared — kernels never ran on the emulator. Results are meaningless."
        return 1
    fi
    if grep -q "Established firmware bundle version\|Mapped hugepage\|KMD version" "$log"; then
        echo "[postflight] INVALID RUN: real-hardware markers present — ran on the physical card, not the emulator."
        return 1
    fi
    local hits
    hits=$(grep -c "\[ASAN ERROR\]" "$log" || true)
    echo "[postflight] VALID emulator run. [ASAN ERROR] count: $hits"
    if [ "$hits" != "0" ]; then
        echo "[postflight] --- ASAN hits ---"
        grep -n "\[ASAN ERROR\]" "$log" | sed 's/^/  /'
    fi
    return 0
}
export -f emule_preflight emule_postflight

# Validate as soon as setup.sh is sourced.
emule_preflight
