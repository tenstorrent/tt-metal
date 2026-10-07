#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Build only the topology qualification executable in a separate build tree.
# Keep the already-qualified Python/native runtime and its build symlink intact.
QWEN_TASK_ROOT=${1:?Provide the isolated Qwen runtime directory}
export CPM_SOURCE_CACHE="$QWEN_TASK_ROOT/cpm-cache"
cmake -S "$QWEN_TASK_ROOT/metal" -B "$QWEN_TASK_ROOT/metal-fabric-build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$QWEN_TASK_ROOT/metal/cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake" \
    -DCMAKE_INSTALL_PREFIX="$QWEN_TASK_ROOT/metal-fabric-install" \
    -DTT_METAL_BUILD_TESTS=ON -DTTNN_BUILD_TESTS=OFF \
    -DWITH_PYTHON_BINDINGS=OFF -DENABLE_TRACY=ON -DTT_UNITY_BUILDS=ON
cmake --build "$QWEN_TASK_ROOT/metal-fabric-build" --target test_tt_fabric --parallel 4
