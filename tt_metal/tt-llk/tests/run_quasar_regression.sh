#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Regression runner for Quasar simulator tests.
# The UMD simulation host lifecycle is managed by pytest (via conftest.py).

set -euo pipefail

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
export CHIP_ARCH=quasar

PYTEST_ARGS=()
TEST_PATTERN="test_*.py"
JOBS=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUASAR_TEST_DIR="${SCRIPT_DIR}/python_tests/quasar"

# ──────────────────────────────────────────────────────────────
# Usage
# ──────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [-- PYTEST_EXTRA_ARGS...]

Runs the Quasar pytest suite against a simulator build.
The simulation host is started and stopped automatically by pytest.

Options:
  -t, --test-pattern GLOB     Test file glob pattern (default: test_*.py)
  -j, --jobs N                Run N xdist workers, one Tensix each (default: 1 process)
  -h, --help                  Show this help message

Required environment variables:
  TT_UMD_SIMULATOR_PATH       Path to the simulator build directory
  NNG_SOCKET_ADDR              NNG socket address (e.g. tcp://host:port)
  NNG_SOCKET_LOCAL_PORT        NNG local port

Examples:
  ./$(basename "$0")
  ./$(basename "$0") -t "test_reduce_*.py" -- -x
  ./$(basename "$0") -j 3
  ./$(basename "$0") -- -v --tb=short
EOF
    exit 0
}

# ──────────────────────────────────────────────────────────────
# Parse CLI arguments
# ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--test-pattern)   TEST_PATTERN="$2";         shift 2 ;;
        -j|--jobs)           JOBS="$2";                 shift 2 ;;
        -h|--help)           usage ;;
        --)                  shift; PYTEST_ARGS+=("$@"); break ;;
        *)                   PYTEST_ARGS+=("$1");        shift ;;
    esac
done

# ──────────────────────────────────────────────────────────────
# Validate required environment variables
# ──────────────────────────────────────────────────────────────
missing=()
for var in TT_UMD_SIMULATOR_PATH NNG_SOCKET_ADDR NNG_SOCKET_LOCAL_PORT; do
    if [[ -z "${!var:-}" ]]; then
        missing+=("$var")
    fi
done

if [[ ${#missing[@]} -gt 0 ]]; then
    echo "ERROR: Required environment variable(s) not set: ${missing[*]}" >&2
    echo "Run with --help for details." >&2
    exit 1
fi

if [[ ! -d "$TT_UMD_SIMULATOR_PATH" ]]; then
    echo "ERROR: Simulator build path does not exist: $TT_UMD_SIMULATOR_PATH" >&2
    exit 1
fi

if [[ -n "$JOBS" ]]; then
    if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: --jobs expects a positive integer, got: $JOBS" >&2
        exit 1
    fi
    PYTEST_ARGS=(-n "$JOBS" "${PYTEST_ARGS[@]+"${PYTEST_ARGS[@]}"}")
fi

if ! command -v pytest &>/dev/null; then
    echo "ERROR: pytest not found in PATH" >&2
    exit 1
fi

if [[ ! -d "$QUASAR_TEST_DIR" ]]; then
    echo "ERROR: Quasar test directory not found: $QUASAR_TEST_DIR" >&2
    exit 1
fi

# ──────────────────────────────────────────────────────────────
# Run pytest (simulation host lifecycle handled by conftest.py)
# ──────────────────────────────────────────────────────────────
echo "============================================================"
echo " Quasar LLK Regression Runner"
echo "============================================================"
echo " Simulator build      : $TT_UMD_SIMULATOR_PATH"
echo " NNG_SOCKET_ADDR      : $NNG_SOCKET_ADDR"
echo " NNG_SOCKET_LOCAL_PORT: $NNG_SOCKET_LOCAL_PORT"
echo " Parallel jobs        : ${JOBS:-<single process>}"
echo " Test pattern         : $TEST_PATTERN"
echo " Extra pytest args    : ${PYTEST_ARGS[*]:-<none>}"
echo "============================================================"
echo ""

pytest_exit=0
(
    cd "$QUASAR_TEST_DIR"
    pytest \
        --run-simulator \
        -x \
        $TEST_PATTERN \
        "${PYTEST_ARGS[@]+"${PYTEST_ARGS[@]}"}"
) || pytest_exit=$?

echo ""
if [[ $pytest_exit -eq 0 ]]; then
    echo "============================================================"
    echo " ALL TESTS PASSED"
    echo "============================================================"
else
    echo "============================================================"
    echo " TESTS FAILED (exit code: $pytest_exit)"
    echo "============================================================"
fi

exit "$pytest_exit"
