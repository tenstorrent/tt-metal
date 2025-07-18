#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e  # Exit immediately if a command exits with a non-zero status
set -o pipefail  # Fail if any command in a pipeline fails

ARCH=$1
case "$ARCH" in
    wormhole)
        ARCH_LLK_ROOT="tt_llk_wormhole_b0"
        ARCH_DEFINE="ARCH_WORMHOLE"
        CHIP_ARCH="wormhole"
        ;;
    blackhole)
        ARCH_LLK_ROOT="tt_llk_blackhole"
        ARCH_DEFINE="ARCH_BLACKHOLE"
        CHIP_ARCH="blackhole"
        ;;
    *)
        echo "Usage: $0 [wormhole|blackhole]"
        exit 1
        ;;
esac

ROOT_DIR=$(git rev-parse --show-toplevel)

cat > "$ROOT_DIR/compile_flags.txt" <<EOF
-D$ARCH_DEFINE
-DTENSIX_FIRMWARE
-DCOMPILE_FOR_TRISC
-std=c++17
-nostdinc++
-nostdinc

-DLLK_TRISC_UNPACK
-DLLK_TRISC_MATH
-DLLK_TRISC_PACK

-isystem
$ROOT_DIR/tests/sfpi/compiler/lib/gcc/riscv32-tt-elf/12.4.0/include/
-isystem
$ROOT_DIR/tests/sfpi/compiler/riscv32-tt-elf/include
-isystem
$ROOT_DIR/tests/sfpi/compiler/riscv32-tt-elf/include/c++/12.4.0
-isystem
$ROOT_DIR/tests/sfpi/compiler/riscv32-tt-elf/include/c++/12.4.0/riscv32-tt-elf
-isystem
$ROOT_DIR/tests/sfpi/include
-I$ROOT_DIR/tests/firmware/riscv/common
-I$ROOT_DIR/tests/firmware/riscv/$CHIP_ARCH
-I$ROOT_DIR/tests/hw_specific/$CHIP_ARCH/inc
-I$ROOT_DIR/$ARCH_LLK_ROOT/common/inc
-I$ROOT_DIR/$ARCH_LLK_ROOT/common/inc/sfpu
-I$ROOT_DIR/$ARCH_LLK_ROOT/llk_lib
-I$ROOT_DIR/tests/helpers/include
EOF

(pkill clangd && clangd >/dev/null 2>&1 &) || true  # restart clang if it's running
