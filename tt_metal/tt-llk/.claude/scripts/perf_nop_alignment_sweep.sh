#!/usr/bin/env bash
# Is the matmul timing bistability triggered by instruction layout?
#
# The probe ladder showed that two timestamps OUTSIDE the pack loop -- about seven
# cycles on a 10,352-cycle run -- remove the second state, while 128 timestamps
# bring a (different) second state back.  No "added work" model explains that.
# What every instrumented build does have in common is a changed code layout.
#
# This sweep changes only the layout.  It puts N nops before the pack loop and
# nothing else: no profiler edits, no extra zones, no timestamps.  Measurement uses
# the TILE_LOOP zone the kernel already has.  N nops cost N cycles for the WHOLE
# run, so any change in behaviour is layout, not time.
set -uo pipefail
LLK=~/tt-metal/tt_metal/tt-llk; PT=$LLK/tests/python_tests; SRC=$LLK/tests/sources
OUT="${OUT:-$HOME/nopsweep}"
RUNS="${RUNS:-40}"; IDX="${IDX:-5742}"; LF="${LF:-64}"
NOPS="${NOPS:-0 1 2 3 4 6 8 12 16}"
export RUNNER_TEMP="${RUNNER_TEMP:-$HOME/llk-wh-build}"

mkdir -p "$OUT"; cd "$PT"; source "$LLK/tests/.venv/bin/activate"
say() { echo "=== $* -- $(date -u +%H:%M:%SZ) ==="; }
restore() { cd "$PT"; git checkout -- perf_math_matmul.py helpers/profiler.py \
            "$SRC/math_matmul_perf.cpp" 2>/dev/null; }
trap 'restore; echo "=== restored ==="' EXIT

git diff --quiet -- perf_math_matmul.py helpers/profiler.py "$SRC/math_matmul_perf.cpp" \
  || { echo "FATAL: tree dirty"; exit 1; }

say "resetting card"; tt-smi -r 2>&1 | tail -2; sleep 10

OBJDUMP=""
for c in riscv32-tt-elf-objdump riscv32-unknown-elf-objdump riscv64-unknown-elf-objdump \
         llvm-objdump objdump; do
    command -v "$c" >/dev/null 2>&1 && { OBJDUMP="$c"; break; }
done
[ -n "$OBJDUMP" ] && say "using $OBJDUMP" || say "no objdump found; skipping disassembly"

dump_pack_elf() {
    local NAME=$1 ELF
    [ -n "$OBJDUMP" ] || return 0
    ELF=$(find "$RUNNER_TEMP" -name pack.elf -printf '%T@ %p\n' 2>/dev/null \
          | sort -rn | head -1 | cut -d" " -f2-)
    [ -n "$ELF" ] || { echo "  (no pack.elf found)"; return 0; }
    "$OBJDUMP" -d "$ELF" > "$OUT/${NAME}_pack.asm" 2>/dev/null \
        && echo "  disassembly -> ${NAME}_pack.asm  ($(wc -l < "$OUT/${NAME}_pack.asm") lines)"
}

run_pass() {
    local N=$1 NAME="nop$1"
    say "pass $NAME  nops=$N"
    restore
    sed -i "s/^            LOOP_FACTOR(1024),\$/            LOOP_FACTOR($LF),/" perf_math_matmul.py
    sed -i "s/^    configuration\.run(perf_report)\$/    configuration.run(perf_report, run_count=$RUNS)/" perf_math_matmul.py
    sed -i "s/^@pytest.mark.perf\$/ALL_TEST_PARAMS = [ALL_TEST_PARAMS[$IDX]]\n\n@pytest.mark.perf/" perf_math_matmul.py
    grep -q "LOOP_FACTOR($LF)," perf_math_matmul.py       || { echo "FATAL: loop factor sed"; exit 1; }
    grep -q "run_count=$RUNS" perf_math_matmul.py         || { echo "FATAL: run_count sed"; exit 1; }
    grep -q "ALL_TEST_PARAMS\[$IDX\]" perf_math_matmul.py || { echo "FATAL: config sed"; exit 1; }

python3 - "$PT/helpers/profiler.py" "$SRC/math_matmul_perf.cpp" "$N" <<'PY'
import sys
prof, kern, n = sys.argv[1], sys.argv[2], int(sys.argv[3])

# Host side only: dump the raw profiler frame. The kernel's own zones are untouched.
t = open(prof).read()
a = "def _stats_l1_to_l1(data: ProfilerData) -> pd.DataFrame:\n"
b = a + '''    import os as _os
    _d = _os.environ.get("TS_DUMP")
    if _d:
        _r = data.raw().copy()
        _r.to_csv(_d, mode="a", header=not _os.path.exists(_d), index=False)
'''
assert t.count(a) == 1, "profiler anchor not unique"
open(prof, "w").write(t.replace(a, b))

if n == 0:
    sys.exit(0)

OLD = """            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_packer_wait_for_math_done_();"""
pad = "\n".join('            asm volatile("nop");' for _ in range(n))
NEW = pad + "\n" + OLD
s = open(kern).read()
assert s.count(OLD) == 1, f"kernel anchor matched {s.count(OLD)} times"
open(kern, "w").write(s.replace(OLD, NEW))
PY
    [ $? -eq 0 ] || { echo "FATAL: patch failed"; exit 1; }

    export TS_DUMP="$OUT/${NAME}_profiler.csv"; rm -f "$TS_DUMP"
    rm -rf "$LLK/perf_data"
    CHIP_ARCH=wormhole pytest -q --override-ini=log_cli=false --compile-producer -n 4 \
      -m perf --perf-run-types L1_TO_L1 -k perf_math_matmul . > "$OUT/${NAME}_compile.log" 2>&1
    CHIP_ARCH=wormhole pytest -q --override-ini=log_cli=false --compile-consumer -n 1 \
      -m perf --perf-run-types L1_TO_L1 -k perf_math_matmul . > "$OUT/${NAME}_run.log" 2>&1
    say "pass $NAME rc=$?  rows=$(wc -l < "$TS_DUMP" 2>/dev/null || echo 0)"
    # Keep the perf CSV: it carries TEXT_SIZE, which proves the binary changed.
    rm -rf "$OUT/$NAME"; cp -r "$LLK/perf_data" "$OUT/$NAME" 2>/dev/null
    # Keep the pack disassembly: the loop's real address is what we are sweeping,
    # and text size alone cannot show whether the compiler re-padded.
    dump_pack_elf "$NAME"
}

for n in $NOPS; do run_pass "$n"; done
say DONE
echo
"$LLK/.claude/scripts/perf_nop_alignment_report.py" "$OUT"
