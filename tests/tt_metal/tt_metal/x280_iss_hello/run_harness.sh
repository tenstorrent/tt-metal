#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Build the X280 ISS harness and run C / ELF workloads end-to-end.
#
# Usage:
#   ./run_harness.sh                  # build + run the bundled C benchmarks
#   ./run_harness.sh path/to/foo.c    # compile one C file against the ISS runtime and run
#   ./run_harness.sh path/to/foo.elf  # run a prebuilt ELF
#   ./run_harness.sh --clean

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-$HERE/build}"
OUT="$WORK/out"
PREFIX="$WORK/iss-prefix"
HARNESS_BIN="$WORK/x280_harness"

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

build_harness() {
    say "build x280_harness (host, links Spike)"
    [[ -x "$PREFIX/bin/spike" ]] || die "Spike missing. Run scripts/fetch_iss.sh first."
    mkdir -p "$WORK"
    g++ -std=c++17 -O2 -Wall -Wextra \
        -I "$PREFIX/include" \
        "$HERE/harness/x280_harness.cpp" \
        -o "$HARNESS_BIN" \
        -L "$PREFIX/lib" -Wl,-rpath,"$PREFIX/lib" \
        -lriscv -lfesvr -ldisasm -lsoftfloat -ldl -lpthread
    echo "harness: $HARNESS_BIN"
}

compile_c() {
    local src="$1"
    local name="$2"
    local elf="$OUT/${name}.elf"
    mkdir -p "$OUT"
    "$GCC" "${FLAGS[@]}" -c "$HERE/src/boot.S" -o "$OUT/boot.o"
    "$GCC" "${FLAGS[@]}" -std=gnu11 -c "$HERE/src/htif.c" -o "$OUT/htif.o"
    "$GCC" "${FLAGS[@]}" -std=gnu11 -c "$HERE/src/iss_printf.c" -o "$OUT/iss_printf.o"
    "$GCC" "${FLAGS[@]}" -std=gnu11 -c "$src" -o "$OUT/${name}.o"
    "$GCC" "${FLAGS[@]}" -T "$HERE/ld/x280_iss.ld" -Wl,--no-warn-rwx-segments \
        "$OUT/boot.o" "$OUT/htif.o" "$OUT/iss_printf.o" "$OUT/${name}.o" \
        -o "$elf"
    echo "$elf"
}

run_elf() {
    local elf="$1"
    shift
    local log="${elf%.elf}.log"
    set +e
    "$HARNESS_BIN" "$elf" "$@" >"$log" 2>&1
    local rc=$?
    set -e
    cat "$log"
    echo "harness exit: $rc"
    return "$rc"
}

expect_in() {
    local log="$1"
    local needle="$2"
    if grep -qF "$needle" "$log"; then
        printf '  \033[32m[ ok ]\033[0m %s\n' "$needle"
        return 0
    fi
    printf '  \033[31m[FAIL]\033[0m missing: %s\n' "$needle"
    return 1
}

# --- setup ---
TC="$(find_toolchain)" || die "no riscv64-unknown-elf toolchain.
Set X280_TOOLCHAIN=<dir> or TT_LLM_ENGINE=<tt-llm-engine checkout>."
GCC="$TC/bin/riscv64-unknown-elf-gcc"

say "0. toolchain"
"$GCC" --version | head -1
echo "toolchain: $TC"

say "1. X280 ISS"
"$HERE/scripts/fetch_iss.sh"
if [[ -x "$WORK/dtc/usr/bin/dtc" ]]; then
    export PATH="$WORK/dtc/usr/bin:$PATH"
    export LD_LIBRARY_PATH="$WORK/dtc/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi

build_harness

mkdir -p "$OUT"
ARCH=rv64gcv_zicsr_zifencei
ABI=lp64d
FLAGS=(-march="$ARCH" -mabi="$ABI" -mcmodel=medany -Os -g -ffreestanding
    -fno-builtin -nostdlib -nostartfiles -Wall
    -I "$HERE/src" -I "$HERE/harness")

# Single-target mode: a C file or an ELF.
if [[ $# -ge 1 ]]; then
    target="$1"
    shift
    if [[ "$target" == *.elf ]]; then
        say "run $target"
        run_elf "$target" "$@"
        exit $?
    fi
    if [[ "$target" == *.c ]]; then
        name="$(basename "$target" .c)"
        say "compile $target"
        elf="$(compile_c "$target" "$name")"
        say "run $elf"
        run_elf "$elf" "$@"
        exit $?
    fi
    die "expected a .c or .elf, got: $target"
fi

# --- bundled e2e suite ---
fails=0
pass() { printf '  \033[32m[PASS]\033[0m %s\n' "$1"; }
fail() { printf '  \033[31m[FAIL]\033[0m %s\n' "$1"; fails=$((fails + 1)); }

run_named_c() {
    local src="$1"
    local name="$2"
    local expect="$3"
    say "bench $name"
    local elf
    elf="$(compile_c "$src" "$name")"
    if run_elf "$elf"; then
        if expect_in "$OUT/${name}.log" "$expect"; then
            pass "$name"
        else
            fail "$name (output)"
        fi
    else
        fail "$name (exit $?)"
    fi
}

run_named_c "$HERE/harness/tests/hello.c" hello "hello from x280 harness"
run_named_c "$HERE/harness/tests/arith.c" arith "PASS arith"
run_named_c "$HERE/harness/tests/fib.c" fib "PASS fib"
run_named_c "$HERE/harness/tests/vector.c" vector "PASS vector"

say "bench load_and_check (host file -> guest memory)"
python3 - "$OUT/load.bin" <<'PY'
import sys
path = sys.argv[1]
data = bytearray(256)
magic = 0x48524E53
data[0] = magic & 0xFF
data[1] = (magic >> 8) & 0xFF
data[2] = (magic >> 16) & 0xFF
data[3] = (magic >> 24) & 0xFF
for i in range(4, 256):
    data[i] = i & 0xFF
open(path, "wb").write(data)
PY
elf="$(compile_c "$HERE/harness/tests/load_and_check.c" load_and_check)"
if run_elf "$elf" --load "$OUT/load.bin@0x08140000"; then
    if expect_in "$OUT/load_and_check.log" "PASS load_and_check"; then
        pass "load_and_check"
    else
        fail "load_and_check (output)"
    fi
else
    fail "load_and_check (exit)"
fi

say "bench fill_and_dump (guest memory -> host file)"
elf="$(compile_c "$HERE/harness/tests/fill_and_dump.c" fill_and_dump)"
if run_elf "$elf" --dump "0x08140000+256:$OUT/fill.bin"; then
    python3 - "$OUT/fill.bin" <<'PY'
import sys
data = open(sys.argv[1], "rb").read()
assert len(data) == 256, len(data)
for i, b in enumerate(data):
    expect = (0xA5 ^ i) & 0xFF
    assert b == expect, (i, b, expect)
print("dump bytes match 0xA5^i")
PY
    if expect_in "$OUT/fill_and_dump.log" "PASS fill_and_dump"; then
        pass "fill_and_dump"
    else
        fail "fill_and_dump (output)"
    fi
else
    fail "fill_and_dump (exit)"
fi

say "bench mem_rw (host file -> guest invert -> host file)"
python3 - "$OUT/mem_rw_in.bin" <<'PY'
import sys
data = bytearray(256)
magic = 0x48524E53
data[0] = magic & 0xFF
data[1] = (magic >> 8) & 0xFF
data[2] = (magic >> 16) & 0xFF
data[3] = (magic >> 24) & 0xFF
for i in range(4, 256):
    data[i] = i & 0xFF
open(sys.argv[1], "wb").write(data)
PY
elf="$(compile_c "$HERE/harness/tests/mem_rw.c" mem_rw)"
if run_elf "$elf" --load "$OUT/mem_rw_in.bin@0x08140000" --dump "0x08140000+256:$OUT/mem_rw_out.bin"; then
    python3 - "$OUT/mem_rw_in.bin" "$OUT/mem_rw_out.bin" <<'PY'
import sys
inp = open(sys.argv[1], "rb").read()
out = open(sys.argv[2], "rb").read()
assert len(inp) == len(out) == 256
assert bytes(b ^ 0xFF for b in inp) == out
print("dump is bitwise invert of loaded file")
PY
    if expect_in "$OUT/mem_rw.log" "PASS mem_rw invert"; then
        pass "mem_rw"
    else
        fail "mem_rw (output)"
    fi
else
    fail "mem_rw (exit)"
fi

say "summary"
if ((fails == 0)); then
    printf '\n\033[32mX280 harness e2e: all benchmarks passed.\033[0m\n'
    echo "  harness  $HARNESS_BIN"
    echo "  logs     $OUT/*.log"
    exit 0
fi
printf '\n\033[31m%d benchmark(s) failed.\033[0m\n' "$fails"
exit 1
