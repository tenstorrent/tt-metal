#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Runs static type checks (mypy + basedpyright) over the tt-triage scripts.
# Configuration lives alongside this script: mypy.ini and pyrightconfig.json.
#
# Usage:
#   ./run_type_checks.sh           # run both checkers
#   ./run_type_checks.sh mypy      # run only mypy
#   ./run_type_checks.sh pyright   # run only basedpyright
#
# Exits non-zero if any selected checker reports problems.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

WHICH="${1:-all}"
status=0

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
    mypy)    run_mypy ;;
    pyright) run_pyright ;;
    all)     run_mypy; run_pyright ;;
    *)
        echo "Unknown argument: $WHICH (expected: mypy | pyright | all)" >&2
        exit 2
        ;;
esac

exit "$status"
