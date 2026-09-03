# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Shared setup for ci.sh and focus.sh. Sourced, not executed.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_TESTS="$(cd "$HERE/.." && pwd)"
LLK_ROOT="$(cd "$PYTHON_TESTS/../.." && pwd)"

# ttnop reads the arch as CHIP_ARCH; tt-metal calls it ARCH_NAME.
export CHIP_ARCH="${CHIP_ARCH:-${ARCH_NAME:-wormhole}}"
export LLK_HOME="$LLK_ROOT"
# The plugin and its modules are imported by bare name from the pytest process.
export PYTHONPATH="$HERE:$PYTHON_TESTS${PYTHONPATH:+:$PYTHONPATH}"

# Where a node id is written from, and the flag that makes the sweep reuse the
# ELFs a producer pass built. metal_env changes both.
TESTS_ROOT="$PYTHON_TESTS"
CONSUMER_ARGS=(--compile-consumer)
# The backend is chosen by --metal alone: a stale export must not be able to send
# the sweep down the Metal path without the environment metal_env sets up.
unset TTNOP_METAL

# Quasar LLK tests run through tt-exalens against the simulator, same flags as
# run_quasar_regression.sh. Silicon (WH/BH) leaves this empty.
PYTEST_SIM_ARGS=()
if [[ "${CHIP_ARCH}" == "quasar" ]]; then
    PYTEST_SIM_ARGS=(--run-simulator --port="${EXALENS_PORT:-5556}" )
fi

DEVICE_LOCK="${TTNOP_DEVICE_LOCK:-/tmp/tt-llk-test-$CHIP_ARCH.lock}"
BUILD_LOCK="${TTNOP_BUILD_LOCK:-/tmp/ttnop-build-$CHIP_ARCH.lock}"

lock_report_dir() {
    # Serialize shared paths across branches
    exec 8<"$1"
    flock 8
}

reset_report_dir() {
    local report_dir="$1"
    [[ ! -f "$report_dir/failures.jsonl" ]] || \
        mv -f "$report_dir/failures.jsonl" "$report_dir/failures.jsonl.prev"
    rm -f "$report_dir/skips.jsonl" "$report_dir/report.md" "$report_dir/junit.xml"
}

setup_report_dir() {
    # Call before cd so a relative report dir stays under ttnop/.
    [[ "$REPORT_DIR" = /* ]] || REPORT_DIR="$HERE/$REPORT_DIR"
    mkdir -p "$REPORT_DIR"
    REPORT_DIR="$(cd "$REPORT_DIR" && pwd)"
    export TTNOP_REPORT_DIR="$REPORT_DIR"
    lock_report_dir "$REPORT_DIR"
}

build_scanner() {
    make --silent -C "$HERE" "scan-$CHIP_ARCH"
}

# --metal: point the sweep at a ttnn op test instead of an LLK kernel test. The
# scan, the sites and the cave arithmetic are shared because they are Tensix
# level; the environment is not. See the Metal section of the README.
#
# The cave has to exist in the kernel image before any of this works:
#   make metal_cave     (once, and again after rebuilding the hw_toolchain target)
metal_env() {
    export TTNOP_METAL=1
    export TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$LLK_ROOT/../.." && pwd)}"
    local default_sim="$TT_METAL_HOME/../tt-umd-simulators/build/emu-quasar-1x3"
    local simulator="${TT_METAL_SIMULATOR:-${TT_UMD_SIMULATOR_PATH:-}}"
    if [[ "$CHIP_ARCH" == "quasar" || "${ARCH_NAME:-}" == "quasar" || -n "$simulator" ]] \
        || { [[ -d "$default_sim" ]] && [[ ! -e /dev/tenstorrent ]]; }; then
        export CHIP_ARCH=quasar
        export ARCH_NAME=quasar
        simulator="${simulator:-$default_sim}"
        simulator="${simulator%/}"
        if [[ -d "$simulator" ]]; then
            export TT_METAL_SIMULATOR="$(cd "$simulator" && pwd)"
        elif [[ -f "$simulator" ]]; then
            export TT_METAL_SIMULATOR="$(cd "$(dirname "$simulator")" && pwd)/$(basename "$simulator")"
        else
            echo "ttnop: Quasar simulator not found: $simulator" >&2
            return 4
        fi
        PYTEST_SIM_ARGS=()
    fi
    # ttnn first (so `import ttnn` hits ttnn/ttnn, not the outer namespace), then
    # tools/ (tracy), then what is already there for the plugin, then the repo
    # root for models/ and tests/.
    export PYTHONPATH="$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools:$PYTHONPATH:$TT_METAL_HOME"
    # Prefer build_Release/{tt_metal,ttnn} over the possibly-stale build_Release/lib copies.
    export LD_LIBRARY_PATH="$TT_METAL_HOME/build_Release/tt_metal:$TT_METAL_HOME/build_Release/ttnn:$TT_METAL_HOME/build_Release/tt_stl:$TT_METAL_HOME/build_Release/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    local metal_py="$TT_METAL_HOME/python_env/bin"
    if [[ -x "$metal_py/python3" ]]; then
        export PATH="$metal_py:$PATH"
    else
        echo "ttnop: $metal_py/python3 not found; --metal cannot import ttnn from the LLK venv" >&2
        return 4
    fi
    # Required. Under fast dispatch the image is staged into a DRAM buffer on the
    # first enqueue, so every later poke into the host image is invisible.
    export TT_METAL_SLOW_DISPATCH_MODE="${TT_METAL_SLOW_DISPATCH_MODE:-1}"
    # The scan reads the post-XIP dump metal writes beside each kernel ELF.
    unset TT_METAL_DISABLE_XIP_DUMP

    # A ttnn test path is written from the repo root, and metal JITs its own
    # kernels on the first launch, so there is no producer pass to consume.
    TESTS_ROOT="$TT_METAL_HOME"
    CONSUMER_ARGS=()
    make --silent -C "$HERE" metal_shim
}

# Run pytest over a node-id file (so a huge suite never hits ARG_MAX), under a
# watchdog that resets the card when every worker stops answering and resumes on
# what is left. See supervise.py.
supervise_nodeids() {
    local ids_file="$1"
    shift
    python3 "$HERE/supervise.py" "$ids_file" "$@"
}
