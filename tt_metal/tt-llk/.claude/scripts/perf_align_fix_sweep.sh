#!/usr/bin/env bash
# Can we pin the pack loop's alignment and stop the bistability for every config?
#
# The nop sweep showed the two states appear only when the pack loop sits 8 bytes
# into a 16-byte block. Move it anywhere else and the fast state is unreachable, so
# every run gives the same number. That was one config; this sweeps all 19,920.
#
# ".p2align 4" makes the assembler pad to the next 16-byte boundary, so the loop
# lands at position 0 whatever precedes it. Adding nops after it shifts to 4, 8, 12.
#
# Baseline to beat: the loop-factor sweep flagged 96 configs at loop factor 1024.
set -uo pipefail
LLK=~/tt-metal/tt_metal/tt-llk; PT=$LLK/tests/python_tests; SRC=$LLK/tests/sources
OUT="${OUT:-$HOME/alignfix}"
RUNS="${RUNS:-20}"; LF="${LF:-1024}"
# name:pad_nops_after_p2align   ("none" = no directive at all)
PASSES="${PASSES:-baseline:none align0:0 align12:3}"
export RUNNER_TEMP="${RUNNER_TEMP:-$HOME/llk-wh-build}"

mkdir -p "$OUT"; cd "$PT"; source "$LLK/tests/.venv/bin/activate"
say() { echo "=== $* -- $(date -u +%H:%M:%SZ) ==="; }
restore() { cd "$PT"; git checkout -- perf_math_matmul.py "$SRC/math_matmul_perf.cpp" 2>/dev/null; }

git diff --quiet -- perf_math_matmul.py "$SRC/math_matmul_perf.cpp" \
  || { echo "FATAL: tree dirty"; exit 1; }
# Arm the cleanup only after the check, so aborting cannot revert another run.
trap 'restore; echo "=== restored ==="' EXIT

say "resetting card"; tt-smi -r 2>&1 | tail -2; sleep 10

run_pass() {
    local NAME=${1%%:*} PAD=${1##*:}
    say "pass $NAME  padding=$PAD"
    restore
    sed -i "s/^    configuration\.run(perf_report)\$/    configuration.run(perf_report, run_count=$RUNS)/" perf_math_matmul.py
    grep -q "run_count=$RUNS" perf_math_matmul.py || { echo "FATAL: run_count sed"; exit 1; }
    if [ "$LF" != "1024" ]; then
        sed -i "s/^            LOOP_FACTOR(1024),\$/            LOOP_FACTOR($LF),/" perf_math_matmul.py
    fi
    if [ "$PAD" != "none" ]; then
python3 - "$SRC/math_matmul_perf.cpp" "$PAD" <<'PY'
import sys
kern, pad = sys.argv[1], int(sys.argv[2])
OLD = """            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_packer_wait_for_math_done_();"""
lines = ['            asm volatile(".p2align 4");']
lines += ['            asm volatile("nop");'] * pad
s = open(kern).read()
assert s.count(OLD) == 1, f"kernel anchor matched {s.count(OLD)} times"
open(kern, "w").write(s.replace(OLD, "\n".join(lines) + "\n" + OLD))
PY
        [ $? -eq 0 ] || { echo "FATAL: patch failed"; exit 1; }
    fi
    rm -rf "$LLK/perf_data"
    CHIP_ARCH=wormhole pytest -q --override-ini=log_cli=false --compile-producer -n 10 \
      -m perf --perf-run-types L1_TO_L1 -k perf_math_matmul . > "$OUT/${NAME}_compile.log" 2>&1
    CHIP_ARCH=wormhole pytest -q --override-ini=log_cli=false --compile-consumer -n 15 \
      -m perf --perf-run-types L1_TO_L1 -k perf_math_matmul . > "$OUT/${NAME}_run.log" 2>&1
    say "pass $NAME rc=$?"
    rm -rf "$OUT/$NAME"; cp -r "$LLK/perf_data" "$OUT/$NAME" 2>/dev/null
}

for P in $PASSES; do run_pass "$P"; done
say DONE
echo
"$LLK/.claude/scripts/perf_align_fix_report.py" "$OUT"
