#!/usr/bin/env bash
CSV="generated/profiler/reports/2026_03_11_08_00_02/ops_perf_results_2026_03_11_08_00_02.csv"
PERF="/home/bklockiewicz/tt-metal/python_env/bin/tt-perf-report"
OUTDIR="sweep_perf_reports_tg_less_sync"

mkdir -p "$OUTDIR"

mapfile -t KEYS < <(grep "signpost" "$CSV" | grep "\-start," | sed 's/,.*//' | sort -u | sed 's/-start$//')

for key in "${KEYS[@]}"; do
    outfile="${OUTDIR}/${key}.txt"
    "$PERF" "$CSV" --start-signpost "${key}-start" --end-signpost "${key}-end" > "$outfile" 2>&1
    echo "done: $key"
done
