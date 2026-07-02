#!/usr/bin/env bash
# Sweep a non-recurrent BiLSTM matmul family (gatex | reorder_in | reorder_out): one config per
# tracy process, extract the median MatmulDeviceOperation device-kernel duration per fresh report.
set -u
cd "$(dirname "$0")/../../../.."   # repo root
SHAPE=${1:-gatex}
NCFG=${2:-22}
T=models/experimental/kokoro/perf/test_gatex_matmul_sweep.py::test_gatex_matmul_sweep
for i in $(seq 0 $((NCFG-1))); do
  KOKORO_MM_SHAPE=$SHAPE KOKORO_MM_CFG=$i timeout 300 python -m tracy -r -p --op-support-count 10000 -m pytest "$T" >/tmp/gx_sweep_$i.log 2>&1
  RC=$?
  REP=$(ls -td generated/profiler/reports/*/ 2>/dev/null | head -1)
  LABEL=$(grep -h "idx=$i\]" /tmp/gx_sweep_$i.log | head -1 | sed 's/.*cfg=//')
  python3 - "$REP" "$i" "$LABEL" "$RC" <<'PY'
import csv, sys, statistics, glob, os
rep, idx, label, rc = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
f = glob.glob(os.path.join(rep, "ops_perf_results_*.csv")) if rep else []
if not f:
    print(f"idx={idx:>2} {label:34s} NO REPORT (rc={rc})"); sys.exit()
rows = list(csv.DictReader(open(f[0])))
d = [float(r["DEVICE KERNEL DURATION [ns]"]) for r in rows if r["OP CODE"] == "MatmulDeviceOperation"]
if len(d) < 5:
    print(f"idx={idx:>2} {label:34s} FATAL/NO MATMUL (rc={rc})"); sys.exit()
d = sorted(d)[2:]
print(f"idx={idx:>2} {label:34s} n={len(d):3d} median={statistics.median(d)/1000:6.3f}us min={min(d)/1000:6.3f}us")
PY
done
