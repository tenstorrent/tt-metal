# =============================================================================
# emule_setup.sh — tt-emule + ASAN environment for emulated test runs
#
#   USAGE:  source tt_metal/impl/emulation/emule_setup.sh   (from the tt-metal repo
#           root; source it — do NOT execute — so the exports persist)
#
# Routes tt-metal at the tt-emule software emulator (instead of a physical WH
# card) and arms the host-side ASAN sanitizers. After sourcing, emule_preflight
# runs automatically to confirm the build + environment are emule-ready.
#
# This file only configures emule. Activate your Python venv and set up the
# normal tt-metal environment (e.g. via create_venv.sh) before sourcing it.
# =============================================================================

# Refuse to be executed instead of sourced (exports would be lost).
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "ERROR: source this file, do not execute it:  source tt_metal/impl/emulation/emule_setup.sh"
    exit 1
fi

# Anchors the mock cluster descriptor path below.
export TT_METAL_HOME="$(pwd)"

# Architecture of the emulated target (matches the wormhole mock cluster below).
export ARCH_NAME=wormhole_b0

# --- Route to the tt-emule software emulator (NOT the physical WH card) -------
# Without these, tt-metal defaults to TargetDevice::Hardware and opens the real
# device — emulated_program_runner.cpp and every ASAN sanitizer are bypassed.
export TT_METAL_EMULE_MODE=1
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_MOCK_CLUSTER_DESC_PATH="$TT_METAL_HOME/tt_metal/third_party/umd/tests/cluster_descriptor_examples/wormhole_N150.yaml"

# --- Arm the sanitizers -------------------------------------------------------
export TT_METAL_EMULE_ASAN=1

# --- emule sanity guards ------------------------------------------------------
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
    [ "$ok" = "1" ] || { echo "[preflight] environment NOT ready — fix the above and re-source emule_setup.sh"; return 1; }
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

# Validate as soon as this file is sourced.
emule_preflight
