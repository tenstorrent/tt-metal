#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Run hello world under qemu-system-riscv64 -machine sifive_u.
# Exercises software integration at the real LIM link address (0x08001000).
# Does not cover X280 vector/CEASE/NOC — those need silicon. See ../README.md.
# Usage: ./run_qemu.sh  (run ../build.sh first)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
L2CPU="$(cd "$HERE/.." && pwd)"
WORK="${WORK:-$L2CPU/build}"
OUT="$WORK/out"
QEMU_DIR="$WORK/qemu"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() {
    printf '\033[31mERROR: %s\033[0m\n' "$*" >&2
    exit 1
}

# ---------------------------------------------------------------------------
say "1. qemu-system-riscv64"

# System qemu, or unpack Ubuntu debs into build/qemu/ (no root).
QEMU="$(command -v qemu-system-riscv64 || true)"
QEMU_LIBS=""
if [[ -z "$QEMU" ]]; then
    QEMU="$QEMU_DIR/root/usr/bin/qemu-system-riscv64"
    QEMU_LIBS="$QEMU_DIR/root/usr/lib/x86_64-linux-gnu"
    if [[ ! -x "$QEMU" ]]; then
        echo "no system qemu-system-riscv64; fetching Ubuntu packages into $QEMU_DIR"
        mkdir -p "$QEMU_DIR"
        (
            cd "$QEMU_DIR"
            apt-get download qemu-system-misc qemu-system-data \
                libfdt1 libpmem1 liburing2 libndctl6 libdaxctl1 2>&1 | tail -2
            for d in *.deb; do dpkg-deb -x "$d" root/; done
        )
    fi
fi
[[ -x "$QEMU" ]] || die "could not obtain qemu-system-riscv64"
run_qemu() { LD_LIBRARY_PATH="$QEMU_LIBS" "$QEMU" "$@"; }
run_qemu --version | head -1

# sifive_u L2 LIM must be at 0x08000000 (same as Blackhole X280).
say "2. confirm the emulated L2 LIM covers X280_ACTIVE_FW_LOAD_ADDR"
mtree=$(printf 'info mtree\nquit\n' | run_qemu -machine sifive_u -smp 2 -bios none \
    -display none -monitor stdio -serial null 2>/dev/null | grep -i "l2lim" | head -1)
echo "  $mtree"
[[ -n "$mtree" ]] || die "emulated machine has no l2lim region"
grep -q "^\s*0000000008000000" <<<"$mtree" || die "l2lim is not at 0x08000000"

# ---------------------------------------------------------------------------
say "3. build the X280_QEMU flavor"

TC_GCC="${X280_GCC:-}"
if [[ -z "$TC_GCC" ]]; then
    # Same toolchain discovery as ../build.sh.
    for cand in "${X280_TOOLCHAIN:-}" "${TT_LLM_ENGINE:-}/x280/toolchain" \
        /data/*/tt-llm-engine*/x280/toolchain "$HOME"/tt-llm-engine*/x280/toolchain; do
        [[ -n "$cand" && -x "$cand/bin/riscv64-unknown-elf-gcc" ]] && {
            TC_GCC="$cand/bin/riscv64-unknown-elf-gcc"
            break
        }
    done
    [[ -z "$TC_GCC" ]] && TC_GCC="$(command -v riscv64-unknown-elf-gcc || true)"
fi
[[ -x "$TC_GCC" ]] || die "no riscv64-unknown-elf-gcc; run ../build.sh first"
TC_BIN="$(dirname "$TC_GCC")"

BSP="$WORK/freedom-e-sdk/bsp/tt-x280-lim"
[[ -d "$BSP/install/lib/debug" ]] || die "freedom-metal not built; run ../build.sh first"

QOUT="$WORK/qemu-out"
mkdir -p "$QOUT"

RISCV_ARCH=$(sed -n 's/^RISCV_ARCH = //p' "$L2CPU/bsp/settings.mk")
RISCV_ABI=$(sed -n 's/^RISCV_ABI = //p' "$L2CPU/bsp/settings.mk")
RISCV_CMODEL=$(sed -n 's/^RISCV_CMODEL = //p' "$L2CPU/bsp/settings.mk")
FLAGS=(-march="$RISCV_ARCH" -mabi="$RISCV_ABI" -mcmodel="$RISCV_CMODEL"
    -Os -g -Wall -ffunction-sections -fdata-sections --specs=nano.specs
    -DX280_QEMU -I "$BSP/install/include" -I "$L2CPU/src")

# Same sources/BSP as hardware; -DX280_QEMU adds UART mirror and drops CEASE.
for src in hello_x280_lim.c x280_lim_console.c x280_bringup.c; do
    "$TC_GCC" "${FLAGS[@]}" -std=gnu11 -c "$L2CPU/src/$src" -o "$QOUT/${src%.c}.o"
done
# Boot hart 1: sifive_u hart 0 is E51 (no FP). Blackhole would keep 0.
"$TC_GCC" "${FLAGS[@]}" -Wl,--gc-sections -nostartfiles -nostdlib \
    -Wl,--defsym=__metal_boot_hart=1 \
    -Wl,--defsym=__stack_size=0x8000 -u _printf_float \
    -L "$BSP/install/lib/debug" -T "$BSP/metal.default.lds" \
    "$QOUT/hello_x280_lim.o" "$QOUT/x280_lim_console.o" "$QOUT/x280_bringup.o" \
    -Wl,--start-group -lc -lgcc -lm -lmetal -lmetal-gloss -Wl,--end-group \
    -o "$QOUT/hello_qemu.elf"
"$TC_BIN/riscv64-unknown-elf-objcopy" -O binary "$QOUT/hello_qemu.elf" "$QOUT/hello_qemu.bin"

# Trampoline: stands in for "host releases L2CPU from reset".
"$TC_GCC" -march=rv64imac_zicsr -mabi=lp64 -nostdlib -nostartfiles \
    -Wl,-Ttext=0x80000000 -Wl,--entry=_start \
    "$HERE/trampoline.S" -o "$QOUT/trampoline.elf"

printf 'firmware: %s bytes, entry 0x%s\n' "$(wc -c <"$QOUT/hello_qemu.bin")" \
    "$("$TC_BIN/riscv64-unknown-elf-readelf" -h "$QOUT/hello_qemu.elf" | sed -n 's/.*Entry point address:.*0x//p')"

# ---------------------------------------------------------------------------
say "4. run"

SER="$QOUT/serial.txt"
MON="$QOUT/monitor.txt"
rm -f "$SER" "$MON"

# Read sentinel + console header (host-loader contract). sleep is required:
# without it, monitor quit races the guest and LIM reads back as zero.
{
    sleep 4
    printf 'xp /1xg 0x08100000\n'  # sentinel
    printf 'xp /2xg 0x08101000\n'  # console magic + len/dropped
    printf 'quit\n'
} | timeout 60 env LD_LIBRARY_PATH="$QEMU_LIBS" "$QEMU" \
    -machine sifive_u -smp 2 -bios none -display none \
    -kernel "$QOUT/trampoline.elf" \
    -device loader,file="$QOUT/hello_qemu.bin",addr=0x08001000 \
    -serial "file:$SER" -monitor stdio >"$MON" 2>&1 || true

echo "--- guest serial output ($(wc -c <"$SER" 2>/dev/null || echo 0) bytes) ---"
cat "$SER" 2>/dev/null || true
echo "--- LIM readback via qemu monitor ---"
grep -E "^0x0000000008(1|10)" "$MON" 2>/dev/null || sed -n '/xp /,$p' "$MON" | head -8

# ---------------------------------------------------------------------------
say "5. verify"

fails=0
check() {
    if [[ "$2" == pass ]]; then
        printf '  \033[32m[ ok ]\033[0m %s\n' "$1"
    else
        printf '  \033[31m[FAIL]\033[0m %s\n' "$1"
        fails=$((fails + 1))
    fi
}
has() { grep -qF "$1" "$SER" 2>/dev/null; }

has "Hello, World!" && check "the program printed \"Hello, World!\"" pass ||
    check "the program printed \"Hello, World!\"" fail
has "hello_x280_lim -- freedom-e-sdk on a Blackhole L2CPU SiFive X280" &&
    check "banner printed (newlib printf through freedom-metal stdio)" pass ||
    check "banner printed" fail
has "metal_cpu_get_current_hartid()      = 1" &&
    check "freedom-metal reported the hart it is running on" pass ||
    check "freedom-metal reported the hart it is running on" fail
has "0.3333333333" && check "hardware double divide produced 1.0/3.0 correctly" pass ||
    check "hardware double divide produced 1.0/3.0 correctly" fail
has "_enter (image first instruction)    = 0x08001000" &&
    check "ran from X280_ACTIVE_FW_LOAD_ADDR (0x08001000) in L2 LIM" pass ||
    check "ran from X280_ACTIVE_FW_LOAD_ADDR (0x08001000) in L2 LIM" fail
has "build flavor                        = X280_QEMU" &&
    check "X280_QEMU flavor confirmed at runtime" pass ||
    check "X280_QEMU flavor confirmed at runtime" fail

grep -qi "deadbeefcafebabe" "$MON" &&
    check "sentinel 0xDEADBEEFCAFEBABE present in LIM at 0x08100000" pass ||
    check "sentinel 0xDEADBEEFCAFEBABE present in LIM at 0x08100000" fail
grep -qi "2800c0ffee000280" "$MON" &&
    check "console magic 0x2800C0FFEE000280 present in LIM at 0x08101000" pass ||
    check "console magic 0x2800C0FFEE000280 present in LIM at 0x08101000" fail

echo
if ((fails == 0)); then
    printf '\033[32mThe integrated hello world ran. All checks passed.\033[0m\n'
    echo
    echo "  serial  $SER"
    echo "  monitor $MON"
    echo "  elf     $QOUT/hello_qemu.elf"
    echo
    echo "Software integration only; see script header and ../../README.md."
else
    printf '\033[31m%d check(s) failed.\033[0m\n' "$fails"
    exit 1
fi
