#!/bin/bash
# Turn a Tracy ops CSV into the sheets in this directory.
#
#   ./make_perf_report.sh <ops_perf_results.csv> <n_layers> <model-name>
#
# e.g. ./make_perf_report.sh prof/reports/*/ops_perf_results_*.csv 42 gemma2-9B_1xp150
#
# n_layers is 42 for gemma2-9B and 46 for gemma2-27B.
#
# Produces, next to this script:
#   <model-name>_decode_report.txt     one steady-state decode iteration
#   <model-name>_decode_ops.csv        same, machine readable
#   <model-name>_decode_summary.csv    per-op-category rollup
#   <model-name>_decode_breakdown.png  stacked device-time chart
#   <model-name>_full_report.txt.gz    whole capture (compile + prefill + decode)
#
# Requires: pip install tt-perf-report
set -euo pipefail

if [ $# -ne 3 ]; then
    sed -n '2,10p' "$0"; exit 1
fi

CSV="$1"; LAYERS="$2"; NAME="$3"
OUT="$(cd "$(dirname "$0")" && pwd)"

command -v tt-perf-report >/dev/null || { echo "need: pip install tt-perf-report"; exit 1; }

tt-perf-report --no-color "$CSV" | gzip -9 > "$OUT/${NAME}_full_report.txt.gz"

# Dump report rows first so we can find the decode iteration boundaries by report ID.
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
tt-perf-report --no-color --no-summary --no-stacked-report --csv "$TMP/rows.csv" "$CSV" >/dev/null 2>&1

# A decode step issues exactly one SdpaDecode per layer, so those IDs delimit
# iterations. Take a middle one: the first can carry compile effects and the last
# can be truncated by the capture ending.
RANGE=$(python3 - "$TMP/rows.csv" "$LAYERS" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
layers = int(sys.argv[2])
idc = [c for c in rows[0] if c.strip().upper() == "ID"][0]
opc = [c for c in rows[0] if "OP CODE" in c.upper()][0]
sd = sorted(int(r[idc]) for r in rows if "SdpaDecode" in r[opc])
starts = sd[0::layers]
print(f"{starts[len(starts)//2]}-{starts[len(starts)//2 + 1] - 1}" if len(starts) >= 3 else "")
PY
)

if [ -z "$RANGE" ]; then
    echo "$NAME: full report only -- fewer than 3 decode iterations in the capture."
    echo "         Re-capture with a larger --max_generated_tokens."
    exit 0
fi

tt-perf-report --no-color --id-range "$RANGE" "$CSV" > "$OUT/${NAME}_decode_report.txt" 2>&1
# --stacked-csv takes a basename: the tool appends .csv for the rollup and .png
# for the matching chart.
tt-perf-report --no-color --id-range "$RANGE" --csv "$OUT/${NAME}_decode_ops.csv" \
    --stacked-csv "$OUT/${NAME}_decode_summary" "$CSV" >/dev/null 2>&1

mv -f "$OUT/${NAME}_decode_summary.png" "$OUT/${NAME}_decode_breakdown.png"

echo "$NAME: wrote decode sheets (id-range $RANGE) + full report"
