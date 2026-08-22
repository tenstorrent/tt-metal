#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Fetch/build the X280 ISS. Prefers a vendor binary at $X280_ISS.
# Otherwise builds Spike (riscv-isa-sim) configured as an X280-class
# rv64gcv / VLEN=512 instruction-set simulator.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
WORK="${WORK:-$ROOT/build}"
PREFIX="$WORK/iss-prefix"
SRC="$WORK/riscv-isa-sim"

say() { printf '\n== %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

if [[ -n "${X280_ISS:-}" && -x "${X280_ISS}" ]]; then
    echo "vendor X280 ISS: $X280_ISS"
    exit 0
fi

if [[ -x "$PREFIX/bin/spike" ]]; then
    echo "spike: $PREFIX/bin/spike"
    exit 0
fi

if command -v spike >/dev/null; then
    echo "system spike: $(command -v spike)"
    exit 0
fi

mkdir -p "$WORK"

# dtc is required to configure Spike. Unpack the Ubuntu package locally (no root).
if ! command -v dtc >/dev/null; then
    say "device-tree-compiler"
    DTC_DIR="$WORK/dtc"
    mkdir -p "$DTC_DIR"
    if [[ ! -x "$DTC_DIR/usr/bin/dtc" ]]; then
        (
            cd "$DTC_DIR"
            apt-get download device-tree-compiler 2>&1 | tail -3
            for d in device-tree-compiler_*.deb; do
                dpkg-deb -x "$d" .
            done
        )
    fi
    export PATH="$DTC_DIR/usr/bin:$PATH"
    export LD_LIBRARY_PATH="$DTC_DIR/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi
command -v dtc >/dev/null || die "dtc not available (needed to build Spike)"

say "clone riscv-isa-sim (Spike ISS)"
if [[ ! -d "$SRC/.git" ]]; then
    git clone --depth 1 https://github.com/riscv-software-src/riscv-isa-sim.git "$SRC"
fi

say "configure + build Spike"
mkdir -p "$SRC/build" "$PREFIX"
(
    cd "$SRC/build"
    if [[ ! -f Makefile ]]; then
        ../configure --prefix="$PREFIX"
    fi
    make -j"$(nproc)"
    make install
)

[[ -x "$PREFIX/bin/spike" ]] || die "spike did not install to $PREFIX/bin/spike"
echo "spike: $PREFIX/bin/spike"
