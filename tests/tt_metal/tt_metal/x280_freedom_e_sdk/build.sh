#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Build the freedom-e-sdk + tt-metal X280 demo and verify the artifacts.
#
# Stages:
#   0  locate tt-metal's sfpi toolchain
#   1  toolchain shim so autoconf accepts the triple
#   2  fetch freedom-e-sdk (+ freedom-metal)
#   3  derive a Quasar DM BSP
#   4  build freedom-metal for -mcpu=tt-qsr64-rocc
#   5  build stock freedom-e-sdk software/hello for that BSP  (checkpoint)
#   6  build hello_x280: freedom-metal + tt-metal X280 cache code
#   7  verify
#
# Usage: ./build.sh [--clean]
#
# Nothing is executed: there is no Quasar silicon and no Quasar simulator, so
# stage 7 inspects the ELFs rather than running them. See README.md.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$HERE/../../../.." && pwd)}"
WORK="${WORK:-$HERE/build}"
ESDK="$WORK/freedom-e-sdk"
SHIM="$WORK/shim"
TARGET=tt-quasar-dm
CONFIG=debug

if [[ "${1:-}" == "--clean" ]]; then
    echo "removing $WORK"
    rm -rf "$WORK"
    exit 0
fi

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() {
    printf '\033[31mERROR: %s\033[0m\n' "$*" >&2
    exit 1
}

# ---------------------------------------------------------------------------
say "0. tt-metal sfpi toolchain"

# sfpi lands in runtime/sfpi of a configured tt-metal build. Set SFPI to point
# somewhere else (a sibling checkout, a shared install) if this worktree has not
# been built.
SFPI="${SFPI:-$TT_METAL_HOME/runtime/sfpi/compiler}"
[[ -x "$SFPI/bin/riscv-tt-elf-gcc" ]] ||
    die "no sfpi at $SFPI. Run '$TT_METAL_HOME/install_dependencies.sh --sfpi', or set SFPI=<path>/runtime/sfpi/compiler."

"$SFPI/bin/riscv-tt-elf-gcc" --version | head -1
echo "tt-metal:  $TT_METAL_HOME"

# The ISA and multilib the DM cores are built for. tt_metal/llrt/hal/tt-2xx/
# quasar/qa_hal.cpp uses -mcpu=tt-qsr64-rocc for HalProcessorClassType::DM.
MCPU=tt-qsr64-rocc
echo "-mcpu=$MCPU  ->  $("$SFPI/bin/riscv-tt-elf-gcc" -mcpu=$MCPU -print-multi-directory)"

# ---------------------------------------------------------------------------
say "1. toolchain shim"

# freedom-metal configures with autoconf, and config.sub rejects the triple
# 'riscv-tt-elf' ("machine riscv-tt not recognized"). Rather than patch upstream
# autotools files, expose the same binaries under the triple freedom-e-sdk
# already expects. The target and multilibs are baked into the compiler, so the
# name it is invoked under does not change what it produces.
mkdir -p "$SHIM/bin"
for f in "$SFPI"/bin/riscv-tt-elf-*; do
    ln -sf "$f" "$SHIM/bin/riscv64-unknown-elf-${f##*/riscv-tt-elf-}"
done
echo "$(find "$SHIM/bin" -type l | wc -l) tools linked as riscv64-unknown-elf-*"

# ---------------------------------------------------------------------------
say "2. freedom-e-sdk"

if [[ ! -d "$ESDK/.git" ]]; then
    mkdir -p "$WORK"
    git clone --depth 1 https://github.com/sifive/freedom-e-sdk.git "$ESDK"
fi

if [[ ! -f "$ESDK/freedom-metal/Makefile.am" ]]; then
    git -C "$ESDK" submodule update --init --depth 1 freedom-metal
fi

# The BSP rules in scripts/libmetal.mk name these generator scripts as
# prerequisites of design.dts et al. They only run when the devicetree is newer,
# but make still insists the files exist.
for sub in devicetree-overlay-generator ldscript-generator esdk-settings-generator cmsis-svd-generator; do
    [[ -f "$ESDK/scripts/$sub/"*.py ]] 2>/dev/null ||
        git -C "$ESDK" submodule update --init --depth 1 "scripts/$sub" >/dev/null
done
echo "freedom-e-sdk:  $(git -C "$ESDK" rev-parse --short HEAD)"
echo "freedom-metal:  $(git -C "$ESDK/freedom-metal" rev-parse --short HEAD)"

# ---------------------------------------------------------------------------
say "3. Quasar DM BSP"

# The proper way to get a BSP is to run freedom-devicetree-tools over a
# design.dts (bsp/quasar-dm.dts describes what that would say). Those tools need
# dtc and are not installed, so start from the nearest upstream rv64 target and
# retarget the two things that actually matter: the ISA, and where the program
# lands in memory. What is left over from qemu-sifive-u54 is its peripheral
# description -- a CLINT, a PLIC and a UART at addresses a Quasar DM core does
# not have. Nothing in this demo touches them; see README.md.
BSP="$ESDK/bsp/$TARGET"
rm -rf "$BSP"
cp -r "$ESDK/bsp/qemu-sifive-u54" "$BSP"
cp "$HERE/bsp/settings.mk" "$BSP/settings.mk"

# Link addresses come from tt-metal, not from this script: ask the preprocessor
# what the DM kernel window is, exactly as the linker scripts in
# tt_metal/hw/toolchain do.
read_mem_map() {
    printf '#include "dev_mem_map.h"\n__VALUE__ %s\n' "$1" |
        "$SFPI/bin/riscv-tt-elf-gcc" -mcpu=$MCPU -E -P -x c - \
            -I "$TT_METAL_HOME/tt_metal/hw/inc/internal/tt-2xx/quasar" |
        sed -n 's/^__VALUE__ //p' |
        python3 -c 'import sys; print(eval(sys.stdin.read()))'
}
KERNEL_BASE=$(read_mem_map MEM_KERNEL_BASE)
KERNEL_SIZE=$(read_mem_map MEM_DM_KERNEL_SIZE)
printf 'MEM_KERNEL_BASE     = 0x%08x\n' "$KERNEL_BASE"
printf 'MEM_DM_KERNEL_SIZE  = 0x%08x (%d KB)\n' "$KERNEL_SIZE" "$((KERNEL_SIZE / 1024))"

# u54 puts everything at 0x80000000, which also happens to be out of range for
# medlow-compiled newlib (R_RISCV_HI20 covers +/-2 GiB around 0). Quasar's low
# TL1 addresses have no such problem.
for lds in "$BSP"/*.lds; do
    sed -i -E "s/^(\s*testram \(airwx\) : ORIGIN =) [^,]+, LENGTH = .*/\1 $(printf '0x%08x' "$KERNEL_BASE"), LENGTH = $(printf '0x%08x' "$KERNEL_SIZE")/" "$lds"
done
grep -h "ORIGIN =" "$BSP/metal.default.lds"

# Keep make from trying to regenerate the BSP: design.dts must not look newer
# than the files derived from it.
touch "$BSP/core.dts" && sleep 0.1
touch "$BSP/design.dts" && sleep 0.1
touch "$BSP"/*.lds "$BSP"/metal*.h "$BSP/design.svd" "$BSP/settings.mk"

# ---------------------------------------------------------------------------
say "4. freedom-metal for -mcpu=$MCPU"

make -C "$ESDK" -j"$(nproc)" \
    TARGET="$TARGET" CONFIGURATION="$CONFIG" PROGRAM=hello \
    RISCV_PATH="$SHIM" \
    metal 2>&1 | grep -vE '^(checking|configure:|config\.status:|  )' | tail -5

ls -l "$BSP/install/lib/$CONFIG/"libmetal*.a

# ---------------------------------------------------------------------------
say "5. checkpoint: stock freedom-e-sdk software/hello"

# Unmodified upstream program, unmodified upstream build system, built for a
# Quasar DM core. If this works, freedom-e-sdk's toolchain and link model are
# compatible with the target before any tt-metal code is involved.
make -C "$ESDK" -j"$(nproc)" \
    TARGET="$TARGET" CONFIGURATION="$CONFIG" PROGRAM=hello \
    RISCV_PATH="$SHIM" \
    software 2>&1 | tail -3

STOCK_ELF="$ESDK/software/hello/$CONFIG/hello.elf"
[[ -f "$STOCK_ELF" ]] || die "stock freedom-e-sdk hello did not build"

# ---------------------------------------------------------------------------
say "6. hello_x280"

OUT="$WORK/out"
mkdir -p "$OUT"
GCC="$SHIM/bin/riscv64-unknown-elf-gcc"
GXX="$SHIM/bin/riscv64-unknown-elf-g++"

# Same flags freedom-e-sdk's scripts/standalone.mk derives from bsp/settings.mk.
# shellcheck disable=SC1091
RISCV_ARCH=$(sed -n 's/^RISCV_ARCH = //p' "$HERE/bsp/settings.mk")
RISCV_ABI=$(sed -n 's/^RISCV_ABI = //p' "$HERE/bsp/settings.mk")
RISCV_CMODEL=$(sed -n 's/^RISCV_CMODEL = //p' "$HERE/bsp/settings.mk")

ARCH_FLAGS=(-mcpu="$MCPU" -march="$RISCV_ARCH" -mabi="$RISCV_ABI" -mcmodel="$RISCV_CMODEL")
COMMON_FLAGS=("${ARCH_FLAGS[@]}" -Os -g -ffunction-sections -fdata-sections --specs=nano.specs)

# freedom-metal headers, and this demo's own.
METAL_INC=(-I "$BSP/install/include" -I "$HERE/src")

# tt-metal device headers. This is the same include set
# tt_metal/llrt/hal/tt-2xx/quasar/qa_hal.cpp hands the JIT compiler for a Quasar
# DM kernel, which is why risc_common.h resolves unmodified.
TT_INC=(
    -I "$TT_METAL_HOME/tt_metal"
    -I "$TT_METAL_HOME/tt_metal/api"
    -I "$TT_METAL_HOME/tt_metal/hw/inc"
    -I "$TT_METAL_HOME/tt_metal/hostdevcommon/api"
    -I "$TT_METAL_HOME/tt_metal/tt-llk/common"
    -I "$TT_METAL_HOME/tt_metal/hw/inc/internal"
    -I "$TT_METAL_HOME/tt_metal/hw/inc/internal/tt-2xx"
    -I "$TT_METAL_HOME/tt_metal/hw/inc/internal/tt-2xx/quasar"
    -I "$TT_METAL_HOME/tt_metal/hw/inc/internal/tt-2xx/quasar/noc"
    -I "$TT_METAL_HOME/tt_metal/hw/inc/internal/tt-2xx/quasar/quasar_defines"
    -I "$TT_METAL_HOME/tt_metal/hw/ckernels/quasar/metal/common"
    -I "$TT_METAL_HOME/tt_metal/hw/ckernels/quasar/metal/llk_io"
)

# What risc_common.h's Quasar DM section is gated on, plus the defines the
# tt-metal device headers expect a JIT build to supply.
TT_DEFINES=(
    -DARCH_QUASAR    # selects the Quasar DM cache block in risc_common.h
    -DCOMPILE_FOR_DM # ... which is DM-core only
    -Dquasar
    -DKERNEL_BUILD    # device (not host) path through dprint_common.h
    -DTENSIX_FIRMWARE # keeps <fmt/core.h> out of tensix_types.h
    -DPROCESSOR_INDEX=0
)

echo "--- x280_cache_tt.cc  (includes tt-metal's risc_common.h)"
"$GXX" "${COMMON_FLAGS[@]}" -std=c++20 -fno-exceptions -fno-rtti \
    "${TT_DEFINES[@]}" "${TT_INC[@]}" "${METAL_INC[@]}" \
    -c "$HERE/src/x280_cache_tt.cc" -o "$OUT/x280_cache_tt.o"

echo "--- quasar_tty.c      (freedom-metal stdio -> Tensix L1)"
"$GCC" "${COMMON_FLAGS[@]}" -std=gnu11 "${METAL_INC[@]}" \
    -c "$HERE/src/quasar_tty.c" -o "$OUT/quasar_tty.o"

echo "--- hello_x280.c      (freedom-metal + tt-metal)"
"$GCC" "${COMMON_FLAGS[@]}" -std=gnu11 "${METAL_INC[@]}" \
    -I "$TT_METAL_HOME/tt_metal/hw/inc/internal/tt-2xx/quasar" \
    -c "$HERE/src/hello_x280.c" -o "$OUT/hello_x280.o"

echo "--- link"
"$GXX" "${COMMON_FLAGS[@]}" \
    -Wl,--gc-sections -Wl,-Map,"$OUT/hello_x280.map" \
    -nostartfiles -nostdlib \
    -L "$BSP/install/lib/$CONFIG" \
    -T "$BSP/metal.default.lds" \
    "$OUT/hello_x280.o" "$OUT/quasar_tty.o" "$OUT/x280_cache_tt.o" \
    -Wl,--start-group -lc -lgcc -lm -lmetal -lmetal-gloss -Wl,--end-group \
    -o "$OUT/hello_x280.elf"

X280_ELF="$OUT/hello_x280.elf"
"$SHIM/bin/riscv64-unknown-elf-objdump" -d --demangle "$X280_ELF" >"$OUT/hello_x280.lst"
"$SHIM/bin/riscv64-unknown-elf-size" "$X280_ELF"

# ---------------------------------------------------------------------------
say "7. verify"

NM="$SHIM/bin/riscv64-unknown-elf-nm"
OBJDUMP="$SHIM/bin/riscv64-unknown-elf-objdump"
READELF="$SHIM/bin/riscv64-unknown-elf-readelf"
fails=0
check() {
    if [[ "$2" == pass ]]; then
        printf '  \033[32m[ ok ]\033[0m %s\n' "$1"
    else
        printf '  \033[31m[FAIL]\033[0m %s\n' "$1"
        fails=$((fails + 1))
    fi
}
expect_grep() { # description, pattern, file
    if grep -qE "$2" "$3"; then check "$1" pass; else check "$1" fail; fi
}

# --- the target really is 64-bit RISC-V
fmt=$("$OBJDUMP" -f "$X280_ELF" | sed -n 's/.*file format //p')
[[ "$fmt" == elf64-littleriscv ]] && check "ELF is elf64-littleriscv" pass ||
    check "ELF is elf64-littleriscv (got $fmt)" fail

# --- it is linked where tt-metal loads DM kernels
entry=$((16#$("$READELF" -h "$X280_ELF" | sed -n 's/.*Entry point address:.*0x//p')))
if ((entry >= KERNEL_BASE && entry < KERNEL_BASE + KERNEL_SIZE)); then
    check "$(printf 'entry 0x%08x is inside the DM kernel window' "$entry")" pass
else
    check "$(printf 'entry 0x%08x is inside the DM kernel window' "$entry")" fail
fi

# --- and it fits there
used=$("$SHIM/bin/riscv64-unknown-elf-size" "$X280_ELF" | awk 'NR==2 {print $1+$2+$3}')
if ((used <= KERNEL_SIZE)); then
    check "$(printf 'image %d B fits MEM_DM_KERNEL_SIZE %d B' "$used" "$KERNEL_SIZE")" pass
else
    check "$(printf 'image %d B fits MEM_DM_KERNEL_SIZE %d B' "$used" "$KERNEL_SIZE")" fail
fi

# --- tt-metal's X280 cache code is really in there
expect_grep "tt-metal risc_common.h emitted fence.i (L1 I\$ invalidate)" \
    'fence\.i' "$OUT/hello_x280.lst"

# --- freedom-metal's own cache.c emits the same instruction.
# It hand-encodes CFLUSH.D.L1 with '.insn i 0x73, 0, x0, addr, -0x40'; sfpi's
# objdump decodes that back to the same mnemonic. Same instruction, two
# independent implementations -- this is the compatibility claim in one line.
"$OBJDUMP" -d --demangle "$X280_ELF" |
    awk '/<metal_dcache_l1_flush>:/,/\sret$/' >"$OUT/metal_dcache_l1_flush.lst"
expect_grep "freedom-metal's metal_dcache_l1_flush emits the same tt.cache.cflush.d.l1" \
    'tt\.cache\.cflush\.d\.l1' "$OUT/metal_dcache_l1_flush.lst"
expect_grep "tt-metal's tt_x280_flush_l1_dcache emits it too" \
    'tt\.cache\.cflush\.d\.l1' <("$OBJDUMP" -d "$X280_ELF" | awk '/<tt_x280_flush_l1_dcache>:/,/\sret$/')

# --- freedom-metal is linked in and used
"$NM" "$X280_ELF" >"$OUT/hello_x280.sym"
for sym in metal_cpu_get_current_hartid metal_dcache_l1_flush metal_icache_l1_available; do
    expect_grep "libmetal symbol $sym linked" " T $sym\$" "$OUT/hello_x280.sym"
done

# --- our Tensix L1 console won, not freedom-metal's UART shim
expect_grep "metal_tty_putc resolved" " T metal_tty_putc\$" "$OUT/hello_x280.sym"
if grep -qE " (T|t) nop_putc\$" "$OUT/hello_x280.sym"; then
    check "freedom-metal's nop/UART tty shim was not pulled in" fail
else
    check "freedom-metal's nop/UART tty shim was not pulled in" pass
fi

# --- the linked ELF records the ISA it was actually built for
elf_arch=$("$READELF" -A "$X280_ELF" | sed -n 's/.*Tag_RISCV_arch: "\(.*\)"/\1/p')
echo "  ELF Tag_RISCV_arch: $elf_arch"
if [[ "$elf_arch" == *xttcache* ]]; then
    check "ELF ISA includes xttcache (the CFLUSH.D.L1 family)" pass
else
    check "ELF ISA includes xttcache (the CFLUSH.D.L1 family)" fail
fi
# A stock X280 is RV64GCV. This target has no F/D and no C, so everything in the
# image -- including newlib's printf -- had to come out soft-float and
# uncompressed. If either extension leaked in, the ISA string would say so.
if [[ "$elf_arch" =~ _[fdcv][0-9] ]]; then
    check "ELF ISA has no F/D/C/V (soft-float, uncompressed)" fail
else
    check "ELF ISA has no F/D/C/V (soft-float, uncompressed)" pass
fi

# --- and the stock upstream program built too
[[ -f "$STOCK_ELF" ]] && check "stock freedom-e-sdk software/hello built for $TARGET" pass ||
    check "stock freedom-e-sdk software/hello built for $TARGET" fail

echo
if ((fails == 0)); then
    printf '\033[32mAll checks passed.\033[0m\n'
    echo
    echo "  hello_x280   $X280_ELF"
    echo "  disassembly  $OUT/hello_x280.lst"
    echo "  map          $OUT/hello_x280.map"
    echo "  stock hello  $STOCK_ELF"
    echo
    echo "Not executed: no Quasar silicon and no Quasar simulator. See README.md."
else
    printf '\033[31m%d check(s) failed.\033[0m\n' "$fails"
    exit 1
fi
