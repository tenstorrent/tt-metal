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

cat > compile_flags.txt <<EOF
-D$ARCH_DEFINE

-DLLK_TRISC_UNPACK
-DLLK_TRISC_MATH
-DLLK_TRISC_PACK

-I../$ARCH_LLK_ROOT/llk_lib
-I../$ARCH_LLK_ROOT/common/inc
-I../$ARCH_LLK_ROOT/common/inc/sfpu
-Ihw_specific/inc
-Ifirmware/riscv/common
-Ifirmware/riscv/$CHIP_ARCH
-Isfpi/include
-Ihelpers/include
EOF

(pkill clangd && clangd >/dev/null 2>&1 &) || true  # restart clang if it's running
