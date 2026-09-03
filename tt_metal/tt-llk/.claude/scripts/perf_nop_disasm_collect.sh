#!/usr/bin/env bash
# Collect the pack disassembly for each nop count, without touching the board.
#
# The nop sweep assumes N nops before the pack loop move that loop by 4N bytes.
# Only the disassembly can confirm that; the compiler may re-pad instead. This
# script builds each variant with --compile-producer, which needs no device, and
# saves the disassembly.  Run it any time; it uses its own build directory so it
# cannot disturb a measurement in progress.
set -uo pipefail
LLK=~/tt-metal/tt_metal/tt-llk; PT=$LLK/tests/python_tests; SRC=$LLK/tests/sources
OUT="${OUT:-$HOME/nopdisasm}"
IDX="${IDX:-5742}"; LF="${LF:-64}"
NOPS="${NOPS:-0 4 8 12 16 20 24 28 32}"
export RUNNER_TEMP="${RUNNER_TEMP:-$HOME/llk-disasm-build}"

mkdir -p "$OUT"; cd "$PT"; source "$LLK/tests/.venv/bin/activate"
say() { echo "=== $* -- $(date -u +%H:%M:%SZ) ==="; }
restore() { cd "$PT"; git checkout -- perf_math_matmul.py "$SRC/math_matmul_perf.cpp" 2>/dev/null; }

git diff --quiet -- perf_math_matmul.py "$SRC/math_matmul_perf.cpp" \
  || { echo "FATAL: tree dirty"; exit 1; }

# Install the cleanup trap only AFTER the dirty-tree check. If it is installed
# first, aborting on a dirty tree runs the restore and reverts the patches of
# whichever run made the tree dirty -- corrupting a measurement in progress.
trap 'restore; echo "=== restored ==="' EXIT

OBJDUMP=""
for c in "$LLK/tests/sfpi/compiler/bin/riscv-tt-elf-objdump" \
         "$LLK/tests/sfpi/compiler/bin/riscv32-tt-elf-objdump" \
         riscv-tt-elf-objdump riscv32-unknown-elf-objdump; do
    [ -x "$c" ] && { OBJDUMP="$c"; break; }
    command -v "$c" >/dev/null 2>&1 && { OBJDUMP="$c"; break; }
done
[ -n "$OBJDUMP" ] || { echo "FATAL: no RISC-V objdump found"; exit 1; }
say "using $OBJDUMP"

for N in $NOPS; do
    NAME="nop$N"
    restore
    sed -i "s/^            LOOP_FACTOR(1024),\$/            LOOP_FACTOR($LF),/" perf_math_matmul.py
    sed -i "s/^@pytest.mark.perf\$/ALL_TEST_PARAMS = [ALL_TEST_PARAMS[$IDX]]\n\n@pytest.mark.perf/" perf_math_matmul.py
    grep -q "ALL_TEST_PARAMS\[$IDX\]" perf_math_matmul.py || { echo "FATAL: config sed"; exit 1; }
    if [ "$N" -gt 0 ]; then
python3 - "$SRC/math_matmul_perf.cpp" "$N" <<'PY'
import sys
kern, n = sys.argv[1], int(sys.argv[2])
OLD = """            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_packer_wait_for_math_done_();"""
pad = "\n".join('            asm volatile("nop");' for _ in range(n))
s = open(kern).read()
assert s.count(OLD) == 1, f"kernel anchor matched {s.count(OLD)} times"
open(kern, "w").write(s.replace(OLD, pad + "\n" + OLD))
PY
        [ $? -eq 0 ] || { echo "FATAL: patch failed"; exit 1; }
    fi
    rm -rf "$RUNNER_TEMP/tt-llk-build"
    CHIP_ARCH=wormhole pytest -q --override-ini=log_cli=false --compile-producer -n 4 \
      -m perf --perf-run-types L1_TO_L1 -k perf_math_matmul . > "$OUT/${NAME}_compile.log" 2>&1
    ELF=$(find "$RUNNER_TEMP" -name pack.elf -printf '%T@ %p\n' 2>/dev/null \
          | sort -rn | head -1 | cut -d" " -f2-)
    if [ -z "$ELF" ]; then
        say "$NAME: no pack.elf built -- see $OUT/${NAME}_compile.log"
        continue
    fi
    "$OBJDUMP" -d "$ELF" > "$OUT/${NAME}_pack.asm" 2> "$OUT/${NAME}_pack.err"
    if [ -s "$OUT/${NAME}_pack.asm" ]; then
        say "$NAME: $(wc -l < "$OUT/${NAME}_pack.asm") lines"
        rm -f "$OUT/${NAME}_pack.err"
    else
        say "$NAME: DISASSEMBLY FAILED -- $(head -1 "$OUT/${NAME}_pack.err")"
    fi
done
say DONE
echo
"$LLK/.claude/scripts/perf_nop_disasm_report.py" "$OUT"
