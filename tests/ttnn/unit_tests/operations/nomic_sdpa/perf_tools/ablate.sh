#!/usr/bin/env bash
# usage: ablate.sh <shape e.g. 8x512> <config> <buffer> <ABLATE list or "-"> [reps]
# One profiled run; prints the device kernel duration(s) of nomic_sdpa (generic op rows).
cd "$(git rev-parse --show-toplevel)"
source python_env/bin/activate
unset TT_METAL_DPRINT_CORES
abl="$4"; [ "$abl" = "-" ] && abl=""
NOMIC_SDPA_REPS="${5:-1}" NOMIC_SDPA_ABLATE="$abl" NOMIC_SDPA_SHAPES="$1" NOMIC_SDPA_CONFIGS="$2" NOMIC_SDPA_BUFFERS="$3" \
NOMIC_SDPA_RESERVE_KB="${NOMIC_SDPA_RESERVE_KB:-176}" \
  scripts/run_safe_pytest.sh --profile tests/ttnn/unit_tests/operations/nomic_sdpa/test_nomic_sdpa_perf.py > /tmp/nomic_ablate_$$.log 2>&1
csv=$(grep -o 'PROFILER CSV: .*' /tmp/nomic_ablate_$$.log | awk '{print $3}')
if [ -z "$csv" ]; then tail -20 /tmp/nomic_ablate_$$.log; exit 1; fi
python - "$csv" "$1 $2 $3 ablate=[$abl]" <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if "Generic" in r["OP CODE"]]
ns = [float(r["DEVICE KERNEL DURATION [ns]"]) / 1000 for r in rows]
import statistics
print(f"{sys.argv[2]}: us = {[round(x, 1) for x in ns]} median {statistics.median(ns):.1f}")
PY
