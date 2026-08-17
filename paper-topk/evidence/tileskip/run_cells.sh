#!/usr/bin/env bash
# Row-parallel topk_large_indices bench: 5 cells x N trials under Tracy.
# Usage: run_cells.sh <tag> <trials>
# Writes per-run device-kernel-duration medians to $OUTDIR/<tag>.csv
set -u
cd /home/nachiket/tt-metal
source python_env/bin/activate
unset TT_METAL_DPRINT_CORES TT_METAL_WATCHER TT_METAL_DEVICE_PROFILER 2>/dev/null || true

TAG="${1:?tag}"
TRIALS="${2:-3}"
OUTDIR=/tmp/claude-1000/-home-nachiket-tt-metal/9f8f10d4-baba-4138-8904-cb9bdebdbd08/scratchpad/storm/tileskip
OUT="$OUTDIR/${TAG}.csv"
echo "cell,trial,ncalls,median_ns,mean_ns,cores" > "$OUT"

BENCH=tests/ttnn/nightly/unit_tests/operations/experimental/_topk_large_indices_bench.py

run_cell() {
  local name="$1" rows="$2" n="$3" k="$4" valid="$5" trial="$6"
  local env_valid=()
  local log="$OUTDIR/logs/${TAG}_${name}_t${trial}.log"
  mkdir -p "$OUTDIR/logs"
  TOPK_ROWS="$rows" TOPK_W="$n" TOPK_K="$k" TOPK_ITERS=5 TOPK_VALID="${valid}" \
    flock /tmp/tt-device.lock timeout 600 python -m tracy -r -v "$BENCH" > "$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "${name},${trial},RC${rc},NA,NA,NA" >> "$OUT"
    echo "FAIL cell=$name trial=$trial rc=$rc (see $log)"
    return 1
  fi
  local csvfile
  csvfile=$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv 2>/dev/null | head -1)
  python - "$csvfile" "$name" "$trial" >> "$OUT" <<'PYEOF'
import csv, sys, statistics
path, name, trial = sys.argv[1], sys.argv[2], sys.argv[3]
rows = list(csv.DictReader(open(path)))
ops = [r for r in rows if "TopkLargeIndices" in (r.get("OP CODE") or "")]
durs = [int(r["DEVICE KERNEL DURATION [ns]"]) for r in ops if r.get("DEVICE KERNEL DURATION [ns]")]
cores = ops[-1]["CORE COUNT"] if ops and "CORE COUNT" in ops[-1] else "NA"
if len(durs) > 1:
    durs = durs[1:]  # drop warmup (first call, JIT/cache fill)
med = int(statistics.median(durs)) if durs else -1
mean = int(statistics.mean(durs)) if durs else -1
print(f"{name},{trial},{len(durs)},{med},{mean},{cores}")
PYEOF
  tail -1 "$OUT"
}

for t in $(seq 1 "$TRIALS"); do
  run_cell r2_n65536_k32    2   65536  32   ""      "$t"
  run_cell r2_n65536_k512   2   65536  512  ""      "$t"
  run_cell r8_n65536_k32    8   65536  32   ""      "$t"
  run_cell r640_n51200_k1536 640 51200 1536 ""      "$t"
  run_cell r2_n102400_k1536v 2  102400 1536 "56320" "$t"
done
echo "DONE -> $OUT"
