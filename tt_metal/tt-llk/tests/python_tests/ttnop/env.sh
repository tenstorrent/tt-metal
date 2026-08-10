# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Shared setup for ci.sh and focus.sh. Sourced, not executed.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_TESTS="$(cd "$HERE/.." && pwd)"
LLK_ROOT="$(cd "$PYTHON_TESTS/../.." && pwd)"

export CHIP_ARCH="${CHIP_ARCH:-wormhole}"
export LLK_HOME="$LLK_ROOT"
# The plugin and its modules are imported by bare name from the pytest process.
export PYTHONPATH="$HERE:$PYTHON_TESTS${PYTHONPATH:+:$PYTHONPATH}"

# Serialise against other agents driving the same silicon, and against each other.
DEVICE_LOCK="${TTNOP_DEVICE_LOCK:-/tmp/tt-llk-test-$CHIP_ARCH.lock}"
BUILD_LOCK="${TTNOP_BUILD_LOCK:-/tmp/ttnop-build-$CHIP_ARCH.lock}"

build_scanner() {
    make --silent -C "$HERE" scan
}

# Run pytest over a node-id file in-process, so a huge suite never hits ARG_MAX.
run_nodeids() {
    local ids_file="$1"
    shift
    python3 - "$ids_file" "$@" <<'PY'
import sys

import pytest

ids = [line.rstrip("\n") for line in open(sys.argv[1]) if line.strip()]
if not ids:
    print("ttnop: nothing collected")
    sys.exit(0)
sys.exit(pytest.main([*ids, *sys.argv[2:], "-p", "ttnop_plugin", "-p", "no:randomly", "-q"]))
PY
}
