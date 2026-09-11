#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Build the bare-metal RV64 x280 firmware images with clang (no riscv64 GCC needed).
#   build/echo.bin  <- fw.c     (stage 2 echo)
#   build/bw.bin    <- fw_bw.c  (stage 3 TLB-window store bandwidth)
# Both execute from cached GDDR at 0x4000_3000_0000 (dram.ld), matching x280_boot.
#
# Usage: ./build_fw.sh                 # build both images here
#        ./build_fw.sh <src.c> <out>   # build one image (used by sibling examples)
set -euo pipefail
cd "$(dirname "$0")"

CLANG="${CLANG:-$(command -v clang-20 || command -v clang)}"
OBJCOPY="${OBJCOPY:-$(command -v llvm-objcopy-20 || command -v llvm-objcopy)}"
READELF="${READELF:-$(command -v llvm-readelf-20 || command -v llvm-readelf || true)}"

HERE="$(pwd)"
FLAGS=(--target=riscv64-unknown-elf -march=rv64ima_zicsr_zicbom -mabi=lp64 -mcmodel=medany
       -mno-relax -nostdlib -ffreestanding -fno-pic -Os -fuse-ld=lld -Wall
       -I"${HERE}" -T "${HERE}/dram.ld" "${HERE}/start.S")

build_one() {  # <src.c> <out-basename-without-ext> [extra -I dirs...]
    local src="$1" out="$2"; shift 2
    mkdir -p "$(dirname "${out}")"
    "${CLANG}" "${FLAGS[@]}" "$@" "${src}" -o "${out}.elf"
    "${OBJCOPY}" -O binary "${out}.elf" "${out}.bin"
    local entry=""
    if [[ -n "${READELF}" ]]; then entry=" entry $("${READELF}" -h "${out}.elf" | awk '/Entry point/{print $NF}')"; fi
    echo "built ${out}.bin ($(stat -c%s "${out}.bin") bytes${entry})"
}

if [[ $# -ge 2 ]]; then
    src="$(realpath "$1")"; out="$2"; shift 2
    build_one "${src}" "${out}" "$@"
else
    build_one "${HERE}/fw.c" "${HERE}/build/echo"
    build_one "${HERE}/fw_bw.c" "${HERE}/build/bw"
fi
