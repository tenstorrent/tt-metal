#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Runs static type checks (mypy + basedpyright) over the tt-triage scripts.
# Configuration lives alongside this script: mypy.ini and pyrightconfig.json.
#
# The type-check toolchain (checkers + stubs + the runtime deps the checkers
# must follow) is pinned in a single place: requirements-dev.txt. This script
# installs it on demand (via uv if available, else pip) when anything is
# missing from the current environment.
#
# Usage:
#   ./run_type_checks.sh           # run both checkers
#   ./run_type_checks.sh mypy      # run only mypy
#   ./run_type_checks.sh pyright   # run only basedpyright
#
# Environment:
#   SKIP_INSTALL=1   skip the install step (assume the toolchain is present)
#
# Exits non-zero if any selected checker reports problems.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Single source of truth for the type-check toolchain pins.
REQUIREMENTS_DEV="$SCRIPT_DIR/requirements-dev.txt"

# inspector_capnp.pyi is generated at C++ build time and is gitignored, so it
# may be absent on a fresh checkout / CI. These let us regenerate it.
STUB_GENERATOR="$SCRIPT_DIR/../../tt_metal/impl/debug/inspector/generate_rpc_stub.py"
INSPECTOR_SCHEMA="inspector.capnp"     # symlink to the rpc.capnp schema
INSPECTOR_STUB="inspector_capnp.pyi"

WHICH="${1:-all}"
status=0

# Install the dev requirements using uv when available, otherwise pip.
install_dev_requirements() {
    if command -v uv >/dev/null 2>&1; then
        echo "Installing $REQUIREMENTS_DEV with uv"
        uv pip install -r "$REQUIREMENTS_DEV"
    else
        echo "Installing $REQUIREMENTS_DEV with pip"
        python3 -m pip install -r "$REQUIREMENTS_DEV"
    fi
}

# True if the toolchain needed for $WHICH is missing from the environment.
toolchain_missing() {
    # The checkers must be able to import ttexalens to follow its types.
    python3 -c "import ttexalens" >/dev/null 2>&1 || return 0
    if [ "$WHICH" = "mypy" ] || [ "$WHICH" = "all" ]; then
        command -v mypy >/dev/null 2>&1 || return 0
    fi
    if [ "$WHICH" = "pyright" ] || [ "$WHICH" = "all" ]; then
        command -v basedpyright >/dev/null 2>&1 || return 0
    fi
    return 1
}

# Ensure the type-check toolchain is installed (no-op if already present).
ensure_tools() {
    [ "${SKIP_INSTALL:-0}" = "1" ] && return 0
    if toolchain_missing; then
        install_dev_requirements
    fi
}

# Regenerate the (gitignored, build-time) inspector_capnp.pyi from the capnp
# schema so the checkers can resolve the inspector_capnp types on a fresh tree.
# Needs only pycapnp, which requirements-dev.txt provides.
generate_inspector_stub() {
    if [ ! -f "$STUB_GENERATOR" ] || [ ! -e "$INSPECTOR_SCHEMA" ]; then
        echo "warning: cannot regenerate $INSPECTOR_STUB (generator or schema missing)" >&2
        return 0
    fi
    echo "==> regenerating $INSPECTOR_STUB"
    python3 "$STUB_GENERATOR" "$INSPECTOR_SCHEMA" "$INSPECTOR_STUB"
}

# Install the toolchain (if needed) and regenerate generated inputs.
prepare() {
    ensure_tools
    generate_inspector_stub
}

run_mypy() {
    echo "==> mypy"
    if ! mypy --config-file mypy.ini; then
        status=1
    fi
}

run_pyright() {
    echo "==> basedpyright"
    if ! basedpyright; then
        status=1
    fi
}

case "$WHICH" in
    mypy)    prepare; run_mypy ;;
    pyright) prepare; run_pyright ;;
    all)     prepare; run_mypy; run_pyright ;;
    *)
        echo "Unknown argument: $WHICH (expected: mypy | pyright | all)" >&2
        exit 2
        ;;
esac

exit "$status"
