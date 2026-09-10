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
CORE=$PT/helpers/perf/core.py
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
# DUMP=1 also writes one row per (variant, run) with the L1_TO_L1 cycles to
# $OUT/<pass>_runs.csv, so per-config medians can be compared across passes.
DUMP="${DUMP:-0}"
# TESTS: perf test modules to run. Both matmul families flag on Wormhole
# (45 perf_math_matmul rows and 8 perf_matmul rows in the 5-run baseline).
TESTS="${TESTS:-perf_math_matmul}"
PASSES="${PASSES:-baseline:none align0:0 align12:3}"
export RUNNER_TEMP="${RUNNER_TEMP:-$HOME/llk-wh-build}"

mkdir -p "$OUT"; cd "$PT"; source "$LLK/tests/.venv/bin/activate"
say() { echo "=== $* -- $(date -u +%H:%M:%SZ) ==="; }
restore() { cd "$PT"; git checkout -- perf_math_matmul.py perf_matmul.py "$SRC/math_matmul_perf.cpp" "$PACKC" "$CORE" 2>/dev/null; }

git diff --quiet -- perf_math_matmul.py perf_matmul.py "$SRC/math_matmul_perf.cpp" "$PACKC" "$CORE" \
  || { echo "FATAL: tree dirty"; exit 1; }
# Arm the cleanup only after the check, so aborting cannot revert another run.
trap 'restore; echo "=== restored ==="' EXIT

KEXPR=$(echo $TESTS | sed "s/ / or /g")
say "tests: $KEXPR"
say "resetting card"; tt-smi -r 2>&1 | tail -2; sleep 10

run_pass() {
    local NAME=${1%%:*} PAD=${1##*:}
    say "pass $NAME  mode=$MODE  n=$PAD"
    restore
    for M in $TESTS; do
        sed -i "s/^    configuration\.run(perf_report)\$/    configuration.run(perf_report, run_count=$RUNS)/" "$M.py"
        grep -q "run_count=$RUNS" "$M.py" || { echo "FATAL: run_count sed in $M.py"; exit 1; }
    done
    if [ "$DUMP" = 1 ]; then
python3 - "$CORE" <<'PY'
import sys
p = sys.argv[1]; t = open(p).read()
OLD = "            stats_df = get_stats(ProfilerData.concat(variant_raw_data))\n"
NEW = """            _d = __import__("os").environ.get("TS_DUMP")
            if _d and run_type == PerfRunType.L1_TO_L1:
                _raw = ProfilerData.concat(variant_raw_data).raw()
                _tl = _raw[_raw["marker"] == "TILE_LOOP"]
                _s = _tl[(_tl["thread"] == "unpack") & (_tl["type"] == "ZONE_START")].sort_values("run_index")
                _e = _tl[(_tl["thread"] == "pack") & (_tl["type"] == "ZONE_END")].sort_values("run_index")
                if len(_s) and len(_s) == len(_e):
                    pd.DataFrame({"variant_id": self.variant_id, "module": self.test_name, "run_index": _s["run_index"].values,
                                  "cycles": _e["timestamp"].values - _s["timestamp"].values}).to_csv(
                        _d, mode="a", header=not __import__("os").path.exists(_d), index=False)
""" + OLD
assert t.count(OLD) == 1, f"core.py anchor matched {t.count(OLD)} times"
open(p, "w").write(t.replace(OLD, NEW))
PY
        [ $? -eq 0 ] || { echo "FATAL: core.py patch failed"; exit 1; }
        export TS_DUMP="$OUT/${NAME}_runs.csv"; rm -f "$TS_DUMP"
    else
        unset TS_DUMP
    fi
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
      -m perf --perf-run-types L1_TO_L1 -k "$KEXPR" . > "$OUT/${NAME}_compile.log" 2>&1
    CHIP_ARCH=wormhole pytest -q --override-ini=log_cli=false --compile-consumer -n 15 \
      -m perf --perf-run-types L1_TO_L1 -k "$KEXPR" . > "$OUT/${NAME}_run.log" 2>&1
    say "pass $NAME rc=$?  runs_rows=$([ -n "${TS_DUMP:-}" ] && wc -l < "$TS_DUMP" 2>/dev/null || echo -)"
    rm -rf "$OUT/$NAME"; cp -r "$LLK/perf_data" "$OUT/$NAME" 2>/dev/null
}

for P in $PASSES; do run_pass "$P"; done
say DONE
echo
"$LLK/.claude/scripts/perf_align_fix_report.py" "$OUT"
[ "$DUMP" = 1 ] && "$LLK/.claude/scripts/perf_median_gate_report.py" "$OUT"
