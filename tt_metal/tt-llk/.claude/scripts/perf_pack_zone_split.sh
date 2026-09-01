#!/usr/bin/env bash
# Split the pack thread's loop body into three zones and find which one grows
# when the measurement lands in its slow rhythm.
set -uo pipefail
PT=~/tt-metal/tt_metal/tt-llk/tests/python_tests; LLK=~/tt-metal/tt_metal/tt-llk
SRC=$LLK/tests/sources/math_matmul_perf.cpp
OUT=~/zones; export RUNNER_TEMP=$HOME/llk-wh-build
RUNS="${RUNS:-100}"; LF="${LF:-32}"; CFG="${CFG:-matmul_config5888}"
mkdir -p "$OUT"; cd "$PT"; source "$LLK/tests/.venv/bin/activate"
trap 'cd "$LLK"; git checkout -- tests/sources/math_matmul_perf.cpp tests/python_tests/perf_math_matmul.py tests/python_tests/helpers/profiler.py; echo "=== restored ==="' EXIT
say(){ echo "=== $* -- $(date -u +%H:%M:%SZ) ==="; }

cd "$LLK"
git diff --quiet -- tests/sources/math_matmul_perf.cpp tests/python_tests/perf_math_matmul.py tests/python_tests/helpers/profiler.py \
  || { echo "FATAL: sources dirty"; exit 1; }

say "patching kernel"
python3 - "$SRC" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); t = p.read_text()
old = """            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                _llk_packer_wait_for_math_done_();
                for (std::uint32_t tile = 0; tile < CT_DIM * RT_DIM; tile++)
                {
                    _llk_pack_<dest_sync, is_fp32_dest_acc_en>(DST_INDEX + tile, PERF_ADDRESS(PERF_OUTPUT, tile));
                }
                _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
            }"""
new = """            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; loop++)
            {
                {
                    START_PERF_MEASURE("PACK_WAIT")
                    _llk_packer_wait_for_math_done_();
                }
                {
                    START_PERF_MEASURE("PACK_WORK")
                    for (std::uint32_t tile = 0; tile < CT_DIM * RT_DIM; tile++)
                    {
                        _llk_pack_<dest_sync, is_fp32_dest_acc_en>(DST_INDEX + tile, PERF_ADDRESS(PERF_OUTPUT, tile));
                    }
                }
                {
                    START_PERF_MEASURE("PACK_DONE")
                    _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                }
            }"""
assert t.count(old) == 1, "kernel pattern not found exactly once"
p.write_text(t.replace(old, new, 1)); print("kernel patched")
PY

say "patching profiler analysis"
python3 - "$PT/helpers/profiler.py" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); t = p.read_text()
old = "    return _stats_timings(pd.concat(timings, ignore_index=True))"
new = """    result = _stats_timings(pd.concat(timings, ignore_index=True))
    _pack = _stats_thread("PACKZONE", data.pack().raw())
    if not _pack.empty:
        result = pd.merge(result, _pack, on=MARKER, how="outer")
    return result"""
assert t.count(old) == 1
p.write_text(t.replace(old, new, 1)); print("profiler patched")
PY

sed -i "s/^            LOOP_FACTOR(1024),\$/            LOOP_FACTOR($LF),/" "$PT/perf_math_matmul.py"
grep -q "LOOP_FACTOR($LF)," "$PT/perf_math_matmul.py" || { echo "FATAL: sed failed"; exit 1; }

cd "$PT"
tt-smi -r >/dev/null 2>&1; sleep 15
say "compiling"
CHIP_ARCH=wormhole pytest -q --compile-producer -m perf --perf-run-types L1_TO_L1 -k "$CFG" . > "$OUT/compile.log" 2>&1
say "compile rc=$?"

: > "$OUT/data.csv"
for i in $(seq 1 "$RUNS"); do
  tt-smi -r >/dev/null 2>&1; sleep 15
  rm -rf "$LLK/perf_data" "$RUNNER_TEMP/tt-llk-build/temp_perf_data"
  CHIP_ARCH=wormhole pytest -q --compile-consumer -n 1 -m perf --perf-run-types L1_TO_L1 -k "$CFG" . >/dev/null 2>&1
  python3 - "$LLK/perf_data/perf_math_matmul/perf_math_matmul.csv" "$i" >> "$OUT/data.csv" 2>/dev/null <<'PY'
import sys, pandas as pd
d = pd.read_csv(sys.argv[1]); i = sys.argv[2]
def g(marker, col):
    r = d[d['marker'] == marker]
    return float(r.iloc[0][col]) if len(r) and col in d.columns and pd.notna(r.iloc[0][col]) else float('nan')
print(f"{i},{g('TILE_LOOP','mean(L1_TO_L1)'):.0f},"
      f"{g('PACK_WAIT','mean(PACKZONE)'):.2f},{g('PACK_WORK','mean(PACKZONE)'):.2f},"
      f"{g('PACK_DONE','mean(PACKZONE)'):.2f}")
PY
  tail -1 "$OUT/data.csv"
done
say DONE
