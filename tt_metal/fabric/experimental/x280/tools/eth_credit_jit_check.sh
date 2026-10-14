#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Compile the eth_credit E0 and E1 kernels the way the Metal JIT does, with no
# device: tt-metal's active_erisck.cc wrapper around the kernel source, the
# JIT's flags (jit_build/build.cpp, llrt/hal/tt-1xx/blackhole/bh_hal.cpp,
# hal_1xx_common.cpp, impl/kernels/kernel.cpp), and only the include roots
# Metal supplies: tt-metal's own, then the kernel's source directory, because
# EthernetConfig carries no compiler_include_paths.
#
# Each shape and bring-up mode below is compiled for ERISC0 (NoC 0) and ERISC1
# (NoC 1). When a Metal runtime cache holds the weakened active-ERISC firmware
# built from this same tt-metal tree, each object is also linked against it;
# otherwise the link is skipped and the script says so.
#
# Usage: eth_credit_jit_check.sh <SFPI g++> <tt-metal dir> <x280 dir> <scratch dir>
set -euo pipefail

GXX=${1:?SFPI g++}
TTM=$(cd "${2:?tt-metal dir}" && pwd)/
X280=$(cd "${3:?x280 dir}" && pwd)
OUT=${4:?scratch dir}
[[ -x "$GXX" ]] || { echo "FAIL: no JIT compiler at $GXX" >&2; exit 1; }
mkdir -p "$OUT"

# The firmware a kernel links against: a cache entry whose dependency hashes
# name this tt-metal tree.
find_weakened() {
    local name=$1 dir
    for dir in "${HOME}"/.cache/tt-metal-cache/*/firmware/"$name"; do
        [[ -f "$dir/${name}_weakened.elf" ]] || continue
        if grep -qs "${TTM}tt_metal/hw/firmware" "$dir"/*.dephash; then
            echo "$dir/${name}_weakened.elf"
            return
        fi
    done
}

COMMON="-std=c++17 -ftt-nttp -ftt-constinit -ftt-consteval -ftt-no-dyninit -flto=1 -flto-partition=one
 -ffast-math -fno-finite-math-only -fsigned-zeros -fno-associative-math -fno-exceptions -fno-rtti
 -fno-use-cxa-atexit -mcpu=tt-bh -fno-tree-loop-distribute-patterns"
CFLAGS="-Os -MMD -Wall -Werror -Wno-error=deprecated-declarations -Wno-error=multistatement-macros
 -Wno-error=parentheses -Wno-error=unused-but-set-variable -Wno-unused-variable -Wno-unused-function
 --param=min-pagesize=0"
# Device-shape defines normally come from the opened device and reach no
# eth_credit code; plausible values compile.
DEVDEFS="-DNUM_DRAM_BANKS=8 -DNUM_L1_BANKS=128 -DLOG_BASE_2_OF_NUM_DRAM_BANKS=3
 -DLOG_BASE_2_OF_NUM_L1_BANKS=7 -DPCIE_NOC_X=19 -DPCIE_NOC_Y=24 -DTENSIX_FIRMWARE -DLOCAL_MEM_EN=0
 -DARCH_BLACKHOLE -DDISPATCH_MESSAGE_ADDR=0 -DKERNEL_BUILD -DCOMPILE_FOR_ERISC -DERISC -DRISC_B0_HW
 -DENABLE_2_ERISC_MODE -DNOC_MODE=0 -DPROGRAMMABLE_CORE_TYPE=1"
INC=""
for d in . .. "" ttnn ttnn/cpp tt_metal/hw/inc tt_metal/tt-llk/common tt_metal/tt-llk/tools/include \
         tt_metal/hostdevcommon/api tt_metal/api/ tt_metal/hw/ckernels/blackhole/metal/common \
         tt_metal/hw/ckernels/blackhole/metal/llk_io tt_metal/hw/inc/internal/tt-1xx \
         tt_metal/hw/inc/internal/tt-1xx/blackhole tt_metal/hw/inc/internal/tt-1xx/blackhole/blackhole_defines \
         tt_metal/hw/inc/internal/tt-1xx/blackhole/noc tt_metal/tt-llk/tt_llk_blackhole/common/inc \
         tt_metal/tt-llk/tt_llk_blackhole/llk_lib tt_metal/tt-llk/tt_llk_blackhole/llk_lib/hal \
         tt_metal/hw/inc/ethernet tt_metal/hw/firmware/src/tt-1xx; do
    case "$d" in
        .|..) INC+=" -I$d" ;;
        *) INC+=" -I${TTM}$d" ;;
    esac
done
INC+=" -I${X280}/kernels/eth_credit"   # the kernel source directory, appended last as the JIT does

WEAK0=$(find_weakened active_erisc || true)
WEAK1=$(find_weakened subordinate_active_erisc || true)
if [[ -z "$WEAK0" || -z "$WEAK1" ]]; then
    echo "NOTE: no Metal runtime cache for ${TTM}; compiling without linking"
fi

SHAPES=(
    "-DEC_P=2 -DEC_N_LINK=8 -DEC_N_DELIVER=8 -DEC_SLOT_SIZE=4192 -DEC_BATCH=2"
    "-DEC_P=1 -DEC_N_LINK=1 -DEC_N_DELIVER=1 -DEC_SLOT_SIZE=112 -DEC_BATCH=1"
    "-DEC_P=4 -DEC_N_LINK=1024 -DEC_N_DELIVER=1024 -DEC_SLOT_SIZE=65616 -DEC_BATCH=1024"
)
E0_MODES=("" "-DEC_E0_SINK_LOCAL=1" "-DEC_E0_SOURCE_SELF=1 -DEC_TAIL_MARKER=1 -DEC_E0_COUNTER_FIRST_DELAY=100")
E1_MODES=("" "-DEC_E1_SINK_LOCAL=1 -DEC_TAIL_MARKER=1")

built=0
build() {
    local role=$1 aerisc=$2 ld=$3 weak=$4 shape=$5 mode=$6
    local dir="$OUT/$role-$built"
    mkdir -p "$dir"
    printf '#include "%s/kernels/eth_credit/%s_kernel.cpp"\n' "$X280" "$role" > "$dir/kernel_includes.hpp"
    # shellcheck disable=SC2086
    (cd "$dir" && "$GXX" $COMMON $CFLAGS $INC $DEVDEFS -DCOMPILE_FOR_AERISC=$aerisc -DPROCESSOR_INDEX=$aerisc \
        -DPHYSICAL_AERISC_ID=$aerisc -DNOC_INDEX=$aerisc $shape $mode \
        $([[ $aerisc == 0 ]] && echo "-Werror=stack-usage=1912") \
        -c -o kernel.o "${TTM}tt_metal/hw/firmware/src/tt-1xx/active_erisck.cc") ||
        { echo "FAIL: $role compile [$shape $mode]" >&2; exit 1; }
    if [[ -n "$weak" ]]; then
        # shellcheck disable=SC2086
        (cd "$dir" && "$GXX" -Os $COMMON -Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 -nostartfiles \
            -Wno-stringop-overflow -T"${TTM}runtime/hw/toolchain/blackhole/$ld" -Wl,--emit-relocs \
            -Wl,--just-symbols="$weak" "${TTM}runtime/hw/lib/blackhole/substitutes.o" kernel.o -o kernel.elf) ||
            { echo "FAIL: $role link [$shape $mode]" >&2; exit 1; }
    fi
    built=$((built + 1))
}

for shape in "${SHAPES[@]}"; do
    for mode in "${E0_MODES[@]}"; do build e0 0 kernel_main_aerisc.ld "$WEAK0" "$shape" "$mode"; done
    for mode in "${E1_MODES[@]}"; do build e1 1 kernel_subordinate_aerisc.ld "$WEAK1" "$shape" "$mode"; done
done
echo "PASS: $built eth_credit ERISC builds$([[ -n "$WEAK0" ]] && echo ", linked" || echo ", compile only")"
