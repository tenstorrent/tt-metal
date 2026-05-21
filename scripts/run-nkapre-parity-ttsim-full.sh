#!/usr/bin/env bash
# Build tt-metal test targets, then run nkapre parity sweep with craq-sim ttsim.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRAQ_SIM="${CRAQ_SIM:-/data/rsong/craq-sim}"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build_Debug}"
ARCH="${ARCH_NAME:-blackhole}"
SIM_ARCH="${SIM_ARCH:-bh}"
NINJA_J="${NINJA_J:-$(nproc)}"
if [ "${NINJA_J}" -gt 32 ]; then NINJA_J=32; fi

export TT_METAL_HOME="$REPO_ROOT"
export ARCH_NAME="$ARCH"
export PYTHONPATH="${TT_METAL_HOME}${PYTHONPATH:+:$PYTHONPATH}"
export CMAKE_BUILD_PARALLEL_LEVEL="${NINJA_J}"

cd "$REPO_ROOT"
mkdir -p craq-parity-results

echo "=== nkapre parity: build + ttsim (craq-sim) ==="
echo "    TT_METAL_HOME=$TT_METAL_HOME"
echo "    ARCH_NAME=$ARCH_NAME  SIM_ARCH=$SIM_ARCH"
echo "    BUILD_DIR=$BUILD_DIR  NINJA_J=$NINJA_J"

echo "=== [1/2] CMake configure ==="
cmake -B build_Debug -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DTT_METAL_BUILD_TESTS=ON \
    -DTTNN_BUILD_TESTS=ON \
    -DBUILD_PROGRAMMING_EXAMPLES=ON \
    -DENABLE_DISTRIBUTED=ON \
    -DWITH_PYTHON_BINDINGS=ON

echo "=== [2/2] Ninja build test targets (-j${NINJA_J}) ==="
ninja -C build_Debug -j"${NINJA_J}" \
    tt-umd tt_metal ttnn \
    unit_tests_ttnn unit_tests_ttnn_tensor unit_tests_ttnn_ccl \
    unit_tests_ttnn_ccl_multi_tensor unit_tests_ttnn_ccl_ops unit_tests_ttnn_accessor \
    test_ccl_multi_cq_multi_device unit_tests_ttnn_udm \
    distributed_program_dispatch distributed_buffer_rw distributed_eltwise_add distributed_trace_and_events \
    distributed_unit_tests unit_tests_eth unit_tests_dispatch unit_tests_debug_tools \
    fabric_unit_tests test_tt_fabric

ln -sfn build_Debug "$REPO_ROOT/build"

export BUILD_DIR
export CRAQ_SIM
export SIM_ARCH
export PER_CMD_TIMEOUT="${PER_CMD_TIMEOUT:-600}"
export RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/craq-parity-results/ttsim-$(date -u +%Y%m%dT%H%M%SZ)}"

echo "=== Starting ttsim sweep -> $RESULTS_DIR ==="
exec "$REPO_ROOT/scripts/run-nkapre-parity-ttsim-sweep.sh"
