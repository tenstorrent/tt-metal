#!/usr/bin/env bash
# Run the recurrent-matmul program-config sweep: one config per tracy process, extract the
# median MatmulDeviceOperation device-kernel duration from each fresh perf report.
set -u
cd "$(dirname "$0")/../../../.."   # repo root
NCFG=${1:-12}
T=models/experimental/kokoro/perf/test_recurrent_matmul_sweep.py::test_recurrent_matmul_sweep
for i in $(seq 0 $((NCFG-1))); do
  KOKORO_MM_CFG=$i timeout 300 python -m tracy -r -p --op-support-count 10000 -m pytest "$T" >/tmp/mm_sweep_$i.log 2>&1
  REP=$(ls -td generated/profiler/reports/*/ 2>/dev/null | head -1)
  LABEL=$(grep -h "\[SWEEP idx=$i\]" /tmp/mm_sweep_$i.log | head -1 | sed 's/.*cfg=//')
  python3 - "$REP" "$i" "$LABEL" <<'PY'
import csv, sys, statistics
rep, idx, label = sys.argv[1], sys.argv[2], sys.argv[3]
import glob, os
f = glob.glob(os.path.join(rep, "ops_perf_results_*.csv"))
if not f:
    print(f"idx={idx:>2} {label:32s} NO REPORT"); sys.exit()
rows = list(csv.DictReader(open(f[0])))
d = [float(r["DEVICE KERNEL DURATION [ns]"]) for r in rows if r["OP CODE"] == "MatmulDeviceOperation"]
if not d:
    print(f"idx={idx:>2} {label:32s} NO MATMUL"); sys.exit()
# skip first few (warmup / compile), take median of the rest
d = sorted(d)[2:]
print(f"idx={idx:>2} {label:32s} n={len(d):3d} median={statistics.median(d)/1000:6.3f}us min={min(d)/1000:6.3f}us")
PY
done
