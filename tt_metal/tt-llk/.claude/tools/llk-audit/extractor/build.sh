#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Build the llk_extract front end against the system Clang/LLVM (>=18).
# Links the aggregate libclang-cpp shared library (and libLLVM — see the link line).
set -euo pipefail
cd "$(dirname "$0")"

# Probe for a suitable llvm-config (>=18), preferring newer. The header says
# ">=18", so do not hardcode a single version.
if [ -z "${LLVM_CONFIG:-}" ]; then
  for c in llvm-config-20 llvm-config-19 llvm-config-18 llvm-config \
           /usr/lib/llvm-20/bin/llvm-config /usr/lib/llvm-19/bin/llvm-config \
           /usr/lib/llvm-18/bin/llvm-config; do
    if command -v "$c" >/dev/null 2>&1; then LLVM_CONFIG="$(command -v "$c")"; break; fi
  done
fi
[ -n "${LLVM_CONFIG:-}" ] && [ -x "$LLVM_CONFIG" ] || {
  echo "llvm-config (>=18) not found; set LLVM_CONFIG" >&2; exit 1; }

CXX="${CXX:-clang++}"
LIBDIR="$($LLVM_CONFIG --libdir)"

# Prefer unversioned -l<name>; fall back to the HIGHEST versioned .so if no dev
# symlink (sort -V so .so.20 beats .so.9). The trailing `|| true` keeps a no-match
# `ls` from failing the pipeline under `pipefail` and tripping `set -e` (this is the
# command after the final `||`); an empty result then fails loudly at link.
pick_lib() {  # $1 = soname (e.g. clang-cpp), $2 = default -l flag
  if [ -e "$LIBDIR/lib$1.so" ]; then
    echo "$2"
  else
    ls "$LIBDIR"/lib"$1".so.* 2>/dev/null | sort -V | tail -1 || true
  fi
}
CLANG_CPP="$(pick_lib clang-cpp -lclang-cpp)"
LLVM_LIB="$(pick_lib LLVM -lLLVM)"

echo "Using $($LLVM_CONFIG --version) at $LIBDIR"
$CXX -std=c++17 -fno-rtti -O2 \
  $($LLVM_CONFIG --cxxflags) \
  llk_extract.cpp -o llk_extract \
  -L"$LIBDIR" $CLANG_CPP $LLVM_LIB \
  -Wl,-rpath,"$LIBDIR"

echo "Built ./llk_extract"
