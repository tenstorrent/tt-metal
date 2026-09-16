#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Build atomic_bench.c for the X280 ISS and run it.
# Usage: ./run_atomic_bench.sh [--clean]

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

say "0. toolchain"
"$GCC" --version | head -1

say "1. X280 ISS"
"$HERE/scripts/fetch_iss.sh"

mkdir -p "$OUT"
ARCH=rv64gcv_zicsr_zifencei
ABI=lp64d
ITERS="${AB_ISS_ITERS:-5}"
FLAGS=(-march="$ARCH" -mabi="$ABI" -mcmodel=medany -Os -g -ffreestanding
    -fno-builtin -nostdlib -nostartfiles -msmall-data-limit=0 -Wall
    -I "$HERE/src" -I "$HERE/include" -DX280_ISS -DAB_ISS_ITERS="${ITERS}ULL")

say "2. build atomic_bench (linked at 0x08001000, iters=$ITERS)"
"$GCC" "${FLAGS[@]}" -c "$HERE/src/boot_atomic.S" -o "$OUT/boot_atomic.o"
"$GCC" "${FLAGS[@]}" -std=gnu11 -c "$HERE/src/htif.c" -o "$OUT/htif.o"
"$GCC" "${FLAGS[@]}" -std=gnu11 -c "$HERE/src/iss_printf.c" -o "$OUT/iss_printf.o"
"$GCC" "${FLAGS[@]}" -std=gnu11 -c "$HERE/src/iss_host_stub.c" -o "$OUT/iss_host_stub.o"
"$GCC" "${FLAGS[@]}" -std=gnu11 -c "$HERE/src/iss_runtime.c" -o "$OUT/iss_runtime.o"
"$GCC" "${FLAGS[@]}" -std=gnu11 -c "$HERE/src/atomic_bench.c" -o "$OUT/atomic_bench.o"
"$GCC" "${FLAGS[@]}" -T "$HERE/ld/x280_iss.ld" -Wl,-Map,"$OUT/atomic_bench.map" \
    -Wl,--no-warn-rwx-segments \
    "$OUT/boot_atomic.o" "$OUT/htif.o" "$OUT/iss_printf.o" \
    "$OUT/iss_host_stub.o" "$OUT/iss_runtime.o" "$OUT/atomic_bench.o" \
    -o "$OUT/atomic_bench.elf"
"$OBJCOPY" -O binary "$OUT/atomic_bench.elf" "$OUT/atomic_bench.bin"
"$OBJDUMP" -d --demangle "$OUT/atomic_bench.elf" >"$OUT/atomic_bench.lst"

say "3. run on X280 ISS"
LOG="$OUT/atomic_bench.log"
set +e
"$HERE/scripts/x280_iss.sh" "$OUT/atomic_bench.elf" >"$LOG" 2>&1
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
check "reached AB_CONFIG_READY" "AB_CONFIG_READY: config latched on ISS"
check "latched config" "iters=$ITERS op=1"
check "amoadd.d check" "amoadd.d x$ITERS -> counter=$ITERS OK"
check "returned to idle" "atomic_bench: hart0 returned to idle (ISS)"

if ((fails == 0 && rc == 0)); then
    printf '\n\033[32matomic_bench ran on the X280 ISS. All checks passed.\033[0m\n'
    echo "  elf  $OUT/atomic_bench.elf"
    echo "  log  $LOG"
    exit 0
fi
printf '\n\033[31m%d check(s) failed (iss exit %d).\033[0m\n' "$fails" "$rc"
exit 1
