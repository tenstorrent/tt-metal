#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end: fetch the X280 ISS, build hello_x280, run it, verify output.
# Usage: ./run.sh [--clean]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-$HERE/build}"
OUT="$WORK/out"

if [[ "${1:-}" == "--clean" ]]; then
    rm -rf "$WORK"
    echo "cleaned $WORK"
    exit 0
fi

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

find_toolchain() {
    if [[ -n "${X280_TOOLCHAIN:-}" && -x "${X280_TOOLCHAIN}/bin/riscv64-unknown-elf-gcc" ]]; then
        echo "$X280_TOOLCHAIN"
        return
    fi
    if [[ -n "${TT_LLM_ENGINE:-}" && -x "${TT_LLM_ENGINE}/x280/toolchain/bin/riscv64-unknown-elf-gcc" ]]; then
        echo "${TT_LLM_ENGINE}/x280/toolchain"
        return
    fi
    for cand in /data/*/tt-llm-engine*/x280/toolchain "$HOME"/tt-llm-engine*/x280/toolchain; do
        [[ -x "$cand/bin/riscv64-unknown-elf-gcc" ]] && { echo "$cand"; return; }
    done
    if command -v riscv64-unknown-elf-gcc >/dev/null; then
        echo "$(dirname "$(dirname "$(command -v riscv64-unknown-elf-gcc)")")"
        return
    fi
    return 1
}

TC="$(find_toolchain)" || die "no riscv64-unknown-elf toolchain.
Set X280_TOOLCHAIN=<dir> or TT_LLM_ENGINE=<tt-llm-engine checkout>."
GCC="$TC/bin/riscv64-unknown-elf-gcc"
OBJCOPY="$TC/bin/riscv64-unknown-elf-objcopy"
OBJDUMP="$TC/bin/riscv64-unknown-elf-objdump"
READELF="$TC/bin/riscv64-unknown-elf-readelf"

say "0. toolchain"
"$GCC" --version | head -1
echo "toolchain: $TC"

say "1. X280 ISS"
"$HERE/scripts/fetch_iss.sh"

mkdir -p "$OUT"
ARCH=rv64gcv_zicsr_zifencei
ABI=lp64d
FLAGS=(-march="$ARCH" -mabi="$ABI" -mcmodel=medany -Os -g -ffreestanding
    -fno-builtin -nostdlib -nostartfiles -Wall -I "$HERE/src")

say "2. build hello_x280 (linked at 0x08001000)"
"$GCC" "${FLAGS[@]}" -c "$HERE/src/boot.S" -o "$OUT/boot.o"
"$GCC" "${FLAGS[@]}" -std=gnu11 -c "$HERE/src/htif.c" -o "$OUT/htif.o"
"$GCC" "${FLAGS[@]}" -std=gnu11 -c "$HERE/src/hello_x280.c" -o "$OUT/hello_x280.o"
"$GCC" "${FLAGS[@]}" -T "$HERE/ld/x280_iss.ld" -Wl,-Map,"$OUT/hello_x280.map" \
    -Wl,--no-warn-rwx-segments \
    "$OUT/boot.o" "$OUT/htif.o" "$OUT/hello_x280.o" -o "$OUT/hello_x280.elf"
"$OBJCOPY" -O binary "$OUT/hello_x280.elf" "$OUT/hello_x280.bin"
"$OBJDUMP" -d --demangle "$OUT/hello_x280.elf" >"$OUT/hello_x280.lst"

entry=$("$READELF" -h "$OUT/hello_x280.elf" | sed -n 's/.*Entry point address:.*0x//p')
echo "entry: 0x$entry"
[[ "$entry" == "8001000" || "$entry" == "08001000" ]] || \
    die "entry 0x$entry is not X280_ACTIVE_FW_LOAD_ADDR 0x08001000"

say "3. run on X280 ISS"
LOG="$OUT/iss.log"
set +e
"$HERE/scripts/x280_iss.sh" "$OUT/hello_x280.elf" >"$LOG" 2>&1
rc=$?
set -e
cat "$LOG"
echo "iss exit: $rc"

say "4. verify"
fails=0
check() {
    if grep -qF "$2" "$LOG"; then
        printf '  \033[32m[ ok ]\033[0m %s\n' "$1"
    else
        printf '  \033[31m[FAIL]\033[0m %s\n' "$1"
        fails=$((fails + 1))
    fi
}
check "printed Hello, World!" "Hello, World!"
check "misa.V is set" "misa.V (vector) = yes"
check "VLEN=512 (vlenb=64)" "vlenb   = 64 bytes -> VLEN=512"
check "vsetvli e32/m1 vl=16" "vsetvli e32/m1 vl = 16"
check "linked at active-FW address" "_start                   = 0x0000000008001000"
check "ISS checks passed" "The X280 ISS hello world ran. All checks passed."

if ((fails == 0 && rc == 0)); then
    printf '\n\033[32mThe simulated X280 hello world ran. All checks passed.\033[0m\n'
    echo "  elf  $OUT/hello_x280.elf"
    echo "  log  $LOG"
    exit 0
fi
printf '\n\033[31m%d check(s) failed (iss exit %d).\033[0m\n' "$fails" "$rc"
exit 1
