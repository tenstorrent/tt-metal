#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Builds the optional _planar_concat extension consumed by
# models/tt_dit/utils/planar_concat.py. Absence of the .so is not an error:
# the caller falls back to the torch scatter path.

set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SRC_DIR}/build"
PYTHON="${PYTHON:-python3}"

if ! grep -q '\bavx2\b' /proc/cpuinfo; then
    echo "build.sh: this CPU reports no AVX2 support; the extension would fault at runtime" >&2
    exit 1
fi

PY_INCLUDE="$("${PYTHON}" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
NUMPY_INCLUDE="$("${PYTHON}" -c 'import numpy; print(numpy.get_include())')"
EXT_SUFFIX="$("${PYTHON}" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"

mkdir -p "${BUILD_DIR}"
OUT="${BUILD_DIR}/_planar_concat${EXT_SUFFIX}"

# -mavx2 covers the streaming stores in planar_concat.cpp as well as the transposes.
${CXX:-g++} -O3 -std=c++17 -fPIC -shared -mavx2 -fvisibility=hidden \
    -I"${SRC_DIR}" -I"${PY_INCLUDE}" -I"${NUMPY_INCLUDE}" \
    "${SRC_DIR}/transpose_avx2.cpp" \
    "${SRC_DIR}/planar_concat.cpp" \
    "${SRC_DIR}/planar_concat_bindings.cpp" \
    -o "${OUT}"

echo "build.sh: wrote ${OUT}"
