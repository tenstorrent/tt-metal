#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Build the llk_extract front end against the system Clang/LLVM (>=18).
# Links the aggregate libclang-cpp shared library.
set -euo pipefail
cd "$(dirname "$0")"

LLVM_CONFIG="${LLVM_CONFIG:-$(command -v llvm-config-20 || echo /usr/lib/llvm-20/bin/llvm-config)}"
[ -x "$LLVM_CONFIG" ] || LLVM_CONFIG="$(command -v llvm-config)"
[ -x "$LLVM_CONFIG" ] || { echo "llvm-config not found; set LLVM_CONFIG" >&2; exit 1; }

CXX="${CXX:-clang++}"
LIBDIR="$($LLVM_CONFIG --libdir)"

# Prefer unversioned -l; fall back to the versioned .so if no dev symlink.
CLANG_CPP="-lclang-cpp"
[ -e "$LIBDIR/libclang-cpp.so" ] || CLANG_CPP="$(ls "$LIBDIR"/libclang-cpp.so.* | head -1)"
LLVM_LIB="-lLLVM"
[ -e "$LIBDIR/libLLVM.so" ] || LLVM_LIB="$(ls "$LIBDIR"/libLLVM.so.* | head -1)"

echo "Using $($LLVM_CONFIG --version) at $LIBDIR"
$CXX -std=c++17 -fno-rtti -O2 \
  $($LLVM_CONFIG --cxxflags) \
  llk_extract.cpp -o llk_extract \
  -L"$LIBDIR" $CLANG_CPP $LLVM_LIB \
  -Wl,-rpath,"$LIBDIR"

echo "Built ./llk_extract"
