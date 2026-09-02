#!/usr/bin/env bash
# How much added work does it take to suppress the matmul timing bistability?
#
# Full instrumentation (4 timestamps on every pack iteration, ~940 cycles) makes
# the two states collapse into one.  Pure additive overhead cannot do that -- both
# states would simply shift up -- so the added work must be absorbing the latency
# the slow state exposes.  This ladder instruments a growing number of iterations
# and finds where the second state disappears, which measures that latency.
set -uo pipefail
LLK=~/tt-metal/tt_metal/tt-llk; PT=$LLK/tests/python_tests; SRC=$LLK/tests/sources
OUT="${OUT:-$HOME/tsexp3}"
RUNS="${RUNS:-40}"; IDX="${IDX:-5742}"; LF="${LF:-64}"
PROBES="${PROBES:--1 0 8 32}"   # -1 clean, 0 two timestamps outside, N first N iterations
export RUNNER_TEMP="${RUNNER_TEMP:-$HOME/llk-wh-build}"

mkdir -p "$OUT"; cd "$PT"; source "$LLK/tests/.venv/bin/activate"
say() { echo "=== $* -- $(date -u +%H:%M:%SZ) ==="; }
restore() { cd "$PT"; git checkout -- perf_math_matmul.py helpers/profiler.py \
            "$SRC/math_matmul_perf.cpp" 2>/dev/null; }
trap 'restore; echo "=== restored ==="' EXIT

git diff --quiet -- perf_math_matmul.py helpers/profiler.py "$SRC/math_matmul_perf.cpp" \
  || { echo "FATAL: tree dirty"; exit 1; }

say "resetting card"; tt-smi -r 2>&1 | tail -2; sleep 10

run_pass() {
    local PROBE=$1 NAME
    case "$PROBE" in
        -1) NAME=baseline ;;
        0)  NAME=probe0 ;;
        *)  NAME="probe$PROBE" ;;
    esac
    say "pass $NAME  probe=$PROBE"
    restore
    sed -i "s/^            LOOP_FACTOR(1024),\$/            LOOP_FACTOR($LF),/" perf_math_matmul.py
    sed -i "s/^    configuration\.run(perf_report)\$/    configuration.run(perf_report, run_count=$RUNS)/" perf_math_matmul.py
    sed -i "s/^@pytest.mark.perf\$/ALL_TEST_PARAMS = [ALL_TEST_PARAMS[$IDX]]\n\n@pytest.mark.perf/" perf_math_matmul.py
    grep -q "LOOP_FACTOR($LF)," perf_math_matmul.py || { echo "FATAL: loop factor sed"; exit 1; }
    grep -q "run_count=$RUNS" perf_math_matmul.py    || { echo "FATAL: run_count sed"; exit 1; }
    grep -q "ALL_TEST_PARAMS\[$IDX\]" perf_math_matmul.py || { echo "FATAL: config sed"; exit 1; }

python3 - "$PT/helpers/profiler.py" "$SRC/math_matmul_perf.cpp" "$PROBE" <<'PY'
import sys
prof, kern, probe = sys.argv[1], sys.argv[2], int(sys.argv[3])

# Host side: dump the whole raw profiler frame so per-run, per-thread timings survive.
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

BODY = """                _llk_packer_wait_for_math_done_();
                for (std::uint32_t tile = 0; tile < CT_DIM * RT_DIM; tile++)
                {
                    _llk_pack_<dest_sync, is_fp32_dest_acc_en>(DST_INDEX + tile, PERF_ADDRESS(PERF_OUTPUT, tile));
                }
                _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();"""
OLD = ("            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)\n"
       "            {\n" + BODY + "\n            }")
INST = """                { TIMESTAMP("T0_ITER") }
                _llk_packer_wait_for_math_done_();
                { TIMESTAMP("T1_GOT") }
                for (std::uint32_t tile = 0; tile < CT_DIM * RT_DIM; tile++)
                {
                    _llk_pack_<dest_sync, is_fp32_dest_acc_en>(DST_INDEX + tile, PERF_ADDRESS(PERF_OUTPUT, tile));
                }
                { TIMESTAMP("T2_PACKED") }
                _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                { TIMESTAMP("T3_DONE") }"""

if probe < 0:
    sys.exit(0)                       # clean control: host dump only
if probe == 0:
    NEW = ('            { TIMESTAMP("TLOOP_S") }\n' + OLD +
           '\n            { TIMESTAMP("TLOOP_E") }')
else:
    # Two separate loops, so the uninstrumented iterations carry no extra branch.
    NEW = (f"            for (std::uint32_t loop = 0; loop < {probe}; loop++)\n"
           f"            {{\n{INST}\n            }}\n"
           f"            for (std::uint32_t loop = {probe}; loop < LOOP_FACTOR; loop++)\n"
           f"            {{\n{BODY}\n            }}")

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
}

for p in $PROBES; do run_pass "$p"; done
say DONE
echo
"$LLK/.claude/scripts/perf_probe_ladder_report.py" "$OUT"
