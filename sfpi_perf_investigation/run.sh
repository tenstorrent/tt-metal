#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Compile every codegen probe and print Tensix instructions per DEST row.
#
# Locates the sfpi toolchain from $SFPI, else the repo's runtime/sfpi, else
# /opt/tenstorrent/sfpi. tt-metal pins the required version in tt_metal/sfpi-version.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/.." && pwd)

if [[ -z "${SFPI:-}" ]]; then
    for candidate in "$repo/runtime/sfpi" /opt/tenstorrent/sfpi; do
        if [[ -x "$candidate/compiler/bin/riscv-tt-elf-g++" ]]; then
            SFPI=$candidate
            break
        fi
    done
fi

if [[ -z "${SFPI:-}" || ! -x "$SFPI/compiler/bin/riscv-tt-elf-g++" ]]; then
    echo "error: no sfpi toolchain found. Set SFPI=/path/to/sfpi, or run" >&2
    echo "       ./install_dependencies.sh --sfpi from the repo root." >&2
    exit 1
fi

gpp=$SFPI/compiler/bin/riscv-tt-elf-g++
have=$("$gpp" --version | head -1 | sed 's/.*sfpi:\([0-9.]*\).*/\1/')
want=$(sed -n "s/^sfpi_version='\(.*\)'/\1/p" "$repo/tt_metal/sfpi-version" 2>/dev/null || true)

echo "sfpi:  $SFPI"
echo "found: $have${want:+   required by tt_metal/sfpi-version: $want}"
if [[ -n "$want" && "$have" != "$want" ]]; then
    echo "warning: version mismatch — instruction counts may differ from the issue." >&2
fi
echo

out=${TMPDIR:-/tmp}/sfpi_perf_investigation
mkdir -p "$out"

for src in "$here"/*.cc; do
    "$gpp" -O2 -mcpu=tt-bh-tensix -I"$SFPI/include" -std=c++17 \
        -I"$here" -S -o "$out/$(basename "${src%.cc}").s" "$src"
done

python3 "$here/count_instructions.py" "$out"/*.s
