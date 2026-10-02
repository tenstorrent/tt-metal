# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Shared setup for ci.sh and focus.sh. Sourced, not executed.

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TTNOP_DIR="$(cd "$SCRIPTS_DIR/.." && pwd)"
PYTHON_TESTS="$(cd "$TTNOP_DIR/.." && pwd)"
LLK_ROOT="$(cd "$PYTHON_TESTS/../.." && pwd)"
RUNTIME_DIR="$TTNOP_DIR/runtime"

export CHIP_ARCH="${CHIP_ARCH:-wormhole}"
export LLK_HOME="$LLK_ROOT"
# The plugin and its modules are imported by bare name from the pytest process.
export PYTHONPATH="$RUNTIME_DIR:$PYTHON_TESTS${PYTHONPATH:+:$PYTHONPATH}"

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
    [[ "$REPORT_DIR" = /* ]] || REPORT_DIR="$TTNOP_DIR/$REPORT_DIR"
    mkdir -p "$REPORT_DIR"
    REPORT_DIR="$(cd "$REPORT_DIR" && pwd)"
    export TTNOP_REPORT_DIR="$REPORT_DIR"
    lock_report_dir "$REPORT_DIR"
}

build_scanner() {
    make --silent -C "$TTNOP_DIR/native" "scan-$CHIP_ARCH"
}

# Run pytest over a node-id file (so a huge suite never hits ARG_MAX), under a
# watchdog that resets the card when every worker stops answering and resumes on
# what is left. See supervise.py.
supervise_nodeids() {
    local ids_file="$1"
    shift
    python3 "$RUNTIME_DIR/supervise.py" "$ids_file" "$@"
}
