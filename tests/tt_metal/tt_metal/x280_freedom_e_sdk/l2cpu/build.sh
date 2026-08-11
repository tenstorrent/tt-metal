#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Build hello_x280_lim -- freedom-e-sdk on a Blackhole L2CPU SiFive X280 -- and
# verify the artifact against the boot contract tt-llm-engine's loader expects.
#
# Stages:
#   0  locate the riscv64-unknown-elf toolchain
#   1  fetch freedom-e-sdk (+ freedom-metal)
#   2  derive the tt-x280-lim BSP
#   3  build freedom-metal
#   4  build stock freedom-e-sdk software/hello for that BSP  (checkpoint)
#   5  build hello_x280_lim
#   6  verify
#
# Usage: ./build.sh [--clean]
#
# This only builds. It does NOT touch hardware -- see ../README.md, and note the
# Galaxy chassis warning there before loading anything onto a device.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-$HERE/build}"
ESDK="$WORK/freedom-e-sdk"
TARGET=tt-x280-lim
CONFIG=debug

# From tt-llm-engine x280/include/x280.h. Also asserted below.
LIM_BASE=0x08000000
ACTIVE_FW_LOAD_ADDR=0x08001000
ACTIVE_FW_REGION_END=0x08120000
SENTINEL_ADDR=0x08100000

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
say "0. riscv64-unknown-elf toolchain"

# No shim and no patching here: the X280 is a stock SiFive core, so the triple
# freedom-e-sdk defaults to (riscv64-unknown-elf) is the one tt-llm-engine
# already vendors for this hardware. Prefer that exact toolchain so the demo and
# the production firmware are built by the same compiler.
find_toolchain() {
    if [[ -n "${X280_TOOLCHAIN:-}" ]]; then
        echo "$X280_TOOLCHAIN"
        return
    fi
    if [[ -n "${TT_LLM_ENGINE:-}" && -x "$TT_LLM_ENGINE/x280/toolchain/bin/riscv64-unknown-elf-gcc" ]]; then
        echo "$TT_LLM_ENGINE/x280/toolchain"
        return
    fi
    # Any tt-llm-engine checkout that has already run `make -C x280`.
    for cand in /data/*/tt-llm-engine*/x280/toolchain "$HOME"/tt-llm-engine*/x280/toolchain; do
        [[ -x "$cand/bin/riscv64-unknown-elf-gcc" ]] && {
            echo "$cand"
            return
        }
    done
    if command -v riscv64-unknown-elf-gcc >/dev/null; then
        echo "$(dirname "$(dirname "$(command -v riscv64-unknown-elf-gcc)")")"
        return
    fi
    return 1
}

TC="$(find_toolchain)" || die "no riscv64-unknown-elf toolchain found.
Set X280_TOOLCHAIN=<dir> (containing bin/riscv64-unknown-elf-gcc), or set
TT_LLM_ENGINE=<tt-llm-engine checkout> and run 'make -C \$TT_LLM_ENGINE/x280'
once to fetch it. tt-llm-engine pulls it from:
  https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2026.04.05/riscv64-elf-ubuntu-22.04-gcc.tar.xz"

GCC="$TC/bin/riscv64-unknown-elf-gcc"
[[ -x "$GCC" ]] || die "$GCC is not executable"
"$GCC" --version | head -1
echo "toolchain: $TC"
echo "default:   $("$GCC" -Q --help=target | sed -n 's/^  -march=  *//p' | head -1)"

# ---------------------------------------------------------------------------
say "1. freedom-e-sdk"

if [[ ! -d "$ESDK/.git" ]]; then
    mkdir -p "$WORK"
    git clone --depth 1 https://github.com/sifive/freedom-e-sdk.git "$ESDK" 2>&1 | tail -1
fi
[[ -f "$ESDK/freedom-metal/Makefile.am" ]] ||
    git -C "$ESDK" submodule update --init --depth 1 freedom-metal 2>&1 | tail -1

# scripts/libmetal.mk names these as prerequisites of the BSP's generated files.
# They only run when the devicetree is newer, but make insists they exist.
for sub in devicetree-overlay-generator ldscript-generator esdk-settings-generator cmsis-svd-generator; do
    compgen -G "$ESDK/scripts/$sub/*.py" >/dev/null ||
        git -C "$ESDK" submodule update --init --depth 1 "scripts/$sub" >/dev/null 2>&1
done
echo "freedom-e-sdk:  $(git -C "$ESDK" rev-parse --short HEAD)"
echo "freedom-metal:  $(git -C "$ESDK/freedom-metal" rev-parse --short HEAD)"

# ---------------------------------------------------------------------------
say "2. tt-x280-lim BSP"

# Generating a BSP properly means running freedom-devicetree-tools over a
# design.dts; that needs dtc, which is not installed. So derive from the nearest
# upstream target and retarget the two things that matter.
#
# qemu-sifive-u54 is the base because its linker script is single-region -- code
# and data both live in RAM, which is how a LIM-resident firmware runs. (The
# sifive-hifive-unmatched BSP is the closer core match and, notably, already
# declares `lim (airwx) : ORIGIN = 0x8000000, LENGTH = 0x1e0000` -- byte for byte
# the X280's LIM region on Blackhole, because it is the same SiFive core-complex
# convention. But its default script boots from SPI flash at 0x20000000, which
# is not the model here.)
BSP="$ESDK/bsp/$TARGET"
rm -rf "$BSP"
cp -r "$ESDK/bsp/qemu-sifive-u54" "$BSP"
cp "$HERE/bsp/settings.mk" "$BSP/settings.mk"

# Active-FW window from x280.h: [0x08001000, 0x08120000).
LIM_LEN=$((ACTIVE_FW_REGION_END - ACTIVE_FW_LOAD_ADDR))
printf 'active FW window: 0x%08x .. 0x%08x (%d KiB)\n' \
    "$ACTIVE_FW_LOAD_ADDR" "$ACTIVE_FW_REGION_END" "$((LIM_LEN / 1024))"
for lds in "$BSP"/*.lds; do
    sed -i -E "s/^(\s*testram \(airwx\) : ORIGIN =) [^,]+, LENGTH = .*/\1 $(printf '0x%08x' "$ACTIVE_FW_LOAD_ADDR"), LENGTH = $(printf '0x%08x' "$LIM_LEN")/" "$lds"
done
grep -h "ORIGIN =" "$BSP/metal.default.lds"

# The SiFive Feature Disable CSR clear in freedom-metal's entry.S is gated on
# this; it must survive into the derived BSP.
grep -q "__metal_chicken_bit = 1" "$BSP/metal.default.lds" ||
    die "BSP linker script does not PROVIDE __metal_chicken_bit = 1"
echo "__metal_chicken_bit = 1 (SiFive Feature Disable CSR 0x7c1 clear enabled)"

touch "$BSP/core.dts" && sleep 0.1
touch "$BSP/design.dts" && sleep 0.1
touch "$BSP"/*.lds "$BSP"/metal*.h "$BSP/design.svd" "$BSP/settings.mk"

# ---------------------------------------------------------------------------
say "3. freedom-metal"

make -C "$ESDK" -j"$(nproc)" \
    TARGET="$TARGET" CONFIGURATION="$CONFIG" PROGRAM=hello \
    RISCV_PATH="$TC" \
    metal 2>&1 | grep -vE '^(checking|configure:|config\.status:|  )' | tail -3
ls -l "$BSP/install/lib/$CONFIG/"libmetal*.a

# ---------------------------------------------------------------------------
say "4. checkpoint: stock freedom-e-sdk software/hello"

make -C "$ESDK" -j"$(nproc)" \
    TARGET="$TARGET" CONFIGURATION="$CONFIG" PROGRAM=hello \
    RISCV_PATH="$TC" \
    software 2>&1 | tail -2
STOCK_ELF="$ESDK/software/hello/$CONFIG/hello.elf"
[[ -f "$STOCK_ELF" ]] || die "stock freedom-e-sdk hello did not build"

# ---------------------------------------------------------------------------
say "5. hello_x280_lim"

OUT="$WORK/out"
mkdir -p "$OUT"

RISCV_ARCH=$(sed -n 's/^RISCV_ARCH = //p' "$HERE/bsp/settings.mk")
RISCV_ABI=$(sed -n 's/^RISCV_ABI = //p' "$HERE/bsp/settings.mk")
RISCV_CMODEL=$(sed -n 's/^RISCV_CMODEL = //p' "$HERE/bsp/settings.mk")

# Identical in shape to tt-llm-engine x280/Makefile's ARCH_FLAGS/CFLAGS, and to
# what freedom-e-sdk's scripts/standalone.mk derives from settings.mk.
FLAGS=(
    -march="$RISCV_ARCH" -mabi="$RISCV_ABI" -mcmodel="$RISCV_CMODEL"
    -Os -g -Wall -ffunction-sections -fdata-sections --specs=nano.specs
    -I "$BSP/install/include" -I "$HERE/src"
)

for src in hello_x280_lim.c x280_lim_console.c x280_bringup.c; do
    echo "--- $src"
    "$GCC" "${FLAGS[@]}" -std=gnu11 -c "$HERE/src/$src" -o "$OUT/${src%.c}.o"
done

echo "--- link"
# Two link-time settings that are not optional, both found by running this under
# emulation (qemu/run_qemu.sh) rather than by reading code:
#
#   __stack_size  The BSP's linker script defaults to 0x400 = 1 KiB per hart.
#                 newlib's vfprintf needs far more than that; with 1 KiB the
#                 stack silently runs off the bottom, corrupts a saved return
#                 address, and the first printf() "returns" to 0. 0x8000 is what
#                 tt-llm-engine's own x280/ld/x280.ld allocates per hart, so
#                 match it.
#   -u _printf_float
#                 nano.specs drops floating-point support from printf; without
#                 this, "%f" prints nothing. Costs a few KB.
#
# Both are PROVIDE/weak in the BSP, so --defsym wins without editing the BSP.
"$GCC" "${FLAGS[@]}" \
    -Wl,--gc-sections -Wl,-Map,"$OUT/hello_x280_lim.map" \
    -Wl,--defsym=__stack_size=0x8000 -u _printf_float \
    -nostartfiles -nostdlib \
    -L "$BSP/install/lib/$CONFIG" \
    -T "$BSP/metal.default.lds" \
    "$OUT/hello_x280_lim.o" "$OUT/x280_lim_console.o" "$OUT/x280_bringup.o" \
    -Wl,--start-group -lc -lgcc -lm -lmetal -lmetal-gloss -Wl,--end-group \
    -o "$OUT/hello_x280_lim.elf"

ELF="$OUT/hello_x280_lim.elf"
BIN="$OUT/hello_x280_lim.bin"

# The loader consumes a raw binary (x280/host/loader.py::load_binary_at), which
# is what tt-llm-engine's own Makefile produces with objcopy -O binary.
"$TC/bin/riscv64-unknown-elf-objcopy" -O binary "$ELF" "$BIN"
"$TC/bin/riscv64-unknown-elf-objdump" -d --demangle "$ELF" >"$OUT/hello_x280_lim.lst"
"$TC/bin/riscv64-unknown-elf-nm" "$ELF" >"$OUT/hello_x280_lim.sym"
"$TC/bin/riscv64-unknown-elf-size" "$ELF"
printf 'raw binary: %s (%s bytes)\n' "$BIN" "$(wc -c <"$BIN")"

# ---------------------------------------------------------------------------
say "6. verify"

READELF="$TC/bin/riscv64-unknown-elf-readelf"
OBJDUMP="$TC/bin/riscv64-unknown-elf-objdump"
fails=0
check() {
    if [[ "$2" == pass ]]; then
        printf '  \033[32m[ ok ]\033[0m %s\n' "$1"
    else
        printf '  \033[31m[FAIL]\033[0m %s\n' "$1"
        fails=$((fails + 1))
    fi
}
expect_grep() {
    if grep -qE "$2" "$3"; then check "$1" pass; else check "$1" fail; fi
}

# --- 64-bit RISC-V
fmt=$("$OBJDUMP" -f "$ELF" | sed -n 's/.*file format //p')
[[ "$fmt" == elf64-littleriscv ]] && check "ELF is elf64-littleriscv" pass ||
    check "ELF is elf64-littleriscv (got $fmt)" fail

# --- the ISA matches what tt-llm-engine builds X280 firmware for.
# -march=rv64gc expands to rv64imafdc + zicsr/zifencei (+ zca/zcd aliases).
elf_arch=$("$READELF" -A "$ELF" | sed -n 's/.*Tag_RISCV_arch: "\(.*\)"/\1/p')
echo "  ELF Tag_RISCV_arch: $elf_arch"
for ext in rv64i m2p0 a2p1 f2p2 d2p2 c2p0; do
    if [[ "$elf_arch" == *"$ext"* ]]; then
        check "ISA has $ext" pass
    else
        check "ISA has $ext" fail
    fi
done

# --- boot contract: the loader writes the raw .bin to ACTIVE_FW_LOAD_ADDR and
# releases reset, so the first byte of the image must BE the entry point.
entry=$((16#$("$READELF" -h "$ELF" | sed -n 's/.*Entry point address:.*0x//p')))
if ((entry == ACTIVE_FW_LOAD_ADDR)); then
    check "$(printf 'entry 0x%08x == X280_ACTIVE_FW_LOAD_ADDR' "$entry")" pass
else
    check "$(printf 'entry 0x%08x == X280_ACTIVE_FW_LOAD_ADDR 0x%08x' "$entry" "$ACTIVE_FW_LOAD_ADDR")" fail
fi

enter=$(sed -n 's/^0*\([0-9a-f]*\) T _enter$/\1/p' "$OUT/hello_x280_lim.sym" | head -1)
if [[ -n "$enter" ]] && ((16#$enter == ACTIVE_FW_LOAD_ADDR)); then
    check "freedom-metal's _enter is the image's first instruction" pass
else
    check "freedom-metal's _enter is the image's first instruction (at 0x$enter)" fail
fi

# --- it fits the active-FW window and stays clear of the host-visible blocks
end=$(sed -n 's/^0*\([0-9a-f]*\) [ABD] metal_segment_bss_target_end$/\1/p' "$OUT/hello_x280_lim.sym" | head -1)
if [[ -n "$end" ]] && ((16#$end < SENTINEL_ADDR)); then
    check "$(printf 'image ends 0x%08x, below the sentinel at 0x%08x' "$((16#$end))" "$SENTINEL_ADDR")" pass
else
    check "image ends below the sentinel at $(printf '0x%08x' "$SENTINEL_ADDR")" fail
fi
binsz=$(wc -c <"$BIN")
if ((binsz <= LIM_LEN)); then
    check "raw binary $binsz B fits the $((LIM_LEN / 1024)) KiB active-FW window" pass
else
    check "raw binary $binsz B fits the $((LIM_LEN / 1024)) KiB active-FW window" fail
fi

# --- SiFive bring-up steps, from freedom-metal's own entry.S
expect_grep "freedom-metal clears SiFive Feature Disable CSR 0x7c1" \
    'csrwi?\s+0x7c1,\s*(zero|0)' "$OUT/hello_x280_lim.lst"

# --- the one gap we had to fill: mstatus.VS enable for the vector unit
expect_grep "__metal_before_start hook linked" " T __metal_before_start\$" "$OUT/hello_x280_lim.sym"
expect_grep "__metal_before_start writes mstatus (VS enable)" \
    'csrs\s+mstatus' <("$OBJDUMP" -d "$ELF" | awk '/<__metal_before_start>:/,/\sret$/')

# --- CEASE, the SiFive halt instruction tt-llm-engine's entry.S also uses
expect_grep "CEASE (0x30500073) present" '30500073' "$OUT/hello_x280_lim.lst"

# --- console + sentinel are what the existing host tooling reads
expect_grep "metal_tty_putc resolved to the LIM console" " T metal_tty_putc\$" "$OUT/hello_x280_lim.sym"
if grep -qE " (T|t) nop_putc\$" "$OUT/hello_x280_lim.sym"; then
    check "freedom-metal's nop/UART tty shim was not pulled in" fail
else
    check "freedom-metal's nop/UART tty shim was not pulled in" pass
fi

# --- freedom-metal is linked in and used
for sym in metal_cpu_get_current_hartid metal_cpu_get_num_harts; do
    expect_grep "libmetal symbol $sym linked" " T $sym\$" "$OUT/hello_x280_lim.sym"
done

# --- hardware float really is in play (this target is full rv64gc, unlike Quasar DM)
expect_grep "hardware FP instructions present (F/D in use)" \
    '\s(fdiv|fadd|fmul|fsub)\.' "$OUT/hello_x280_lim.lst"

# --- and the stock upstream program built too
[[ -f "$STOCK_ELF" ]] && check "stock freedom-e-sdk software/hello built for $TARGET" pass ||
    check "stock freedom-e-sdk software/hello built for $TARGET" fail

echo
if ((fails == 0)); then
    printf '\033[32mAll checks passed.\033[0m\n'
    echo
    echo "  raw binary   $BIN"
    echo "  elf          $ELF"
    echo "  disassembly  $OUT/hello_x280_lim.lst"
    echo
    echo "Loadable by tt-llm-engine's x280/host/loader.py::load_binary_at at"
    printf '  lim_addr=0x%08x\n' "$ACTIVE_FW_LOAD_ADDR"
    echo "READ ../README.md BEFORE RUNNING ON HARDWARE -- the pyluwen boot path"
    echo "hangs a Galaxy chassis and needs a PSU power cycle to recover."
else
    printf '\033[31m%d check(s) failed.\033[0m\n' "$fails"
    exit 1
fi
