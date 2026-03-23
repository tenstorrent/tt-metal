#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build-standalone"
TT_METALIUM_PREFIX_DEFAULT="${REPO_ROOT}/build/lib/cmake/tt-metalium"
TT_METALIUM_PREFIX="${TT_METALIUM_PREFIX:-${TT_METALIUM_PREFIX_DEFAULT}}"
JOBS="${JOBS:-4}"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DTT_METALIUM_PREFIX="${TT_METALIUM_PREFIX}" -DCMAKE_PREFIX_PATH="${REPO_ROOT}/build;${TT_METALIUM_PREFIX}"
cmake --build "${BUILD_DIR}" --target deepseek_v3_b1_pipeline_manager_core test_pipeline_manager -j "${JOBS}"

echo ""
echo "Built:"
echo "  ${BUILD_DIR}/libdeepseek_v3_b1_pipeline_manager_core.a"
echo "  ${BUILD_DIR}/test_pipeline_manager"
echo ""
echo "Run tests:"
echo "  ${BUILD_DIR}/test_pipeline_manager"
