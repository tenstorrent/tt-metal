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
PACKC=$LLK/tt_llk_wormhole_b0/llk_lib/llk_pack_common.h
OUT="${OUT:-$HOME/alignfix}"
RUNS="${RUNS:-20}"; LF="${LF:-1024}"
# name:N   ("none" = unmodified). What N means depends on MODE:
#   MODE=p2align  ".p2align 4" before the PACK loop, then N nops  (pins pack alignment)
#   MODE=mathnop  N nops inside the MATH loop, after the wait for DEST
#   MODE=thcon    the one LLK change from PR 54157: _llk_pack_dest_section_done_ stalls on
#                PACK | THCON instead of PACK alone. N is ignored. Same instruction size,
#                so the code layout is identical and any change is the stall itself.
# On config 5742 one math nop removed the bistability and ran 9% faster; the compiler
# restructured the math loop (+24 bytes for the first nop). This sweeps all configs.
MODE="${MODE:-p2align}"
PASSES="${PASSES:-baseline:none align0:0 align12:3}"
export RUNNER_TEMP="${RUNNER_TEMP:-$HOME/llk-wh-build}"

mkdir -p "$OUT"; cd "$PT"; source "$LLK/tests/.venv/bin/activate"
say() { echo "=== $* -- $(date -u +%H:%M:%SZ) ==="; }
restore() { cd "$PT"; git checkout -- perf_math_matmul.py "$SRC/math_matmul_perf.cpp" "$PACKC" 2>/dev/null; }

git diff --quiet -- perf_math_matmul.py "$SRC/math_matmul_perf.cpp" "$PACKC" \
  || { echo "FATAL: tree dirty"; exit 1; }
# Arm the cleanup only after the check, so aborting cannot revert another run.
trap 'restore; echo "=== restored ==="' EXIT

say "resetting card"; tt-smi -r 2>&1 | tail -2; sleep 10

run_pass() {
    local NAME=${1%%:*} PAD=${1##*:}
    say "pass $NAME  mode=$MODE  n=$PAD"
    restore
    sed -i "s/^    configuration\.run(perf_report)\$/    configuration.run(perf_report, run_count=$RUNS)/" perf_math_matmul.py
    grep -q "run_count=$RUNS" perf_math_matmul.py || { echo "FATAL: run_count sed"; exit 1; }
    if [ "$LF" != "1024" ]; then
        sed -i "s/^            LOOP_FACTOR(1024),\$/            LOOP_FACTOR($LF),/" perf_math_matmul.py
    fi
    if [ "$PAD" != "none" ]; then
python3 - "$SRC/math_matmul_perf.cpp" "$PAD" "$MODE" "$PACKC" <<'PY'
import sys
kern, pad, mode, packc = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
target = kern
if mode == "thcon":
    target = packc
    OLD = "    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::PACK); // wait for pack to finish"
    NEW = "    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::PACK | p_stall::THCON); // PR 54157: also drain THCON"
elif mode == "mathnop":
    OLD = "                _llk_math_wait_for_dest_available_<dest_sync>();"
    NEW = OLD + "\n" + "\n".join(['                asm volatile("nop");'] * pad)
else:
    OLD = """            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_packer_wait_for_math_done_();"""
    lines = ['            asm volatile(".p2align 4");']
    lines += ['            asm volatile("nop");'] * pad
    NEW = "\n".join(lines) + "\n" + OLD
s = open(target).read()
assert s.count(OLD) == 1, f"anchor matched {s.count(OLD)} times in {target}"
open(target, "w").write(s.replace(OLD, NEW))
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
