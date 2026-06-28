#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# One-shot build for the planar_concat nanobind extension.
#
# Override PYTHON to choose a non-default interpreter:
#     PYTHON=/path/to/venv/bin/python ./build.sh
#
# Override CMAKE_BUILD_TYPE for a debug build:
#     CMAKE_BUILD_TYPE=Debug ./build.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

# Require a real nanobind (cmake_dir), not just an importable name — a namespace
# package on PYTHONPATH can shadow it and make a bare `import nanobind` succeed.
if ! "$PYTHON" -c "import nanobind; nanobind.cmake_dir()" 2>/dev/null; then
    echo "[build.sh] usable nanobind (with cmake_dir) not found in $PYTHON; installing it..."
    "$PYTHON" -m pip install nanobind
fi

PYTHON_ABS="$("$PYTHON" -c 'import sys; print(sys.executable)')"

echo "[build.sh] using python: $PYTHON_ABS"
echo "[build.sh] build type:   $BUILD_TYPE"

cmake -B build -S . \
    -DPython_EXECUTABLE="$PYTHON_ABS" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
cmake --build build -j

echo
echo "[build.sh] success — artefact:"
ls -la build/_planar_concat*.so 2>/dev/null || ls -la build/
