#!/usr/bin/env bash
# Profile main.py under tracy and emit per-signpost tt-perf-report windows.
# Usage: run_profile.sh <label>
set -euo pipefail
LABEL="${1:-run}"
cd /home/mvasiljevic/tt-metal/deepseek_codegen/graph_0
export TT_METAL_HOME=/home/mvasiljevic/tt-metal
export PYTHONPATH=/home/mvasiljevic/tt-metal:/home/mvasiljevic/tt-metal/deepseek_codegen/graph_0
export ARCH_NAME=blackhole TT_METAL_CCACHE_KERNEL_SUPPORT=1
PY=/home/mvasiljevic/tt-metal/python_env/bin/python

echo "[profile] tracy run start"
$PY -m tracy -r -p main.py
# tracy writes the report under $TT_METAL_HOME/generated, not graph_0/generated
REPDIR="$TT_METAL_HOME/generated/profiler/reports"
TS=$(ls -t "$REPDIR" | head -1)
CSV="$REPDIR/$TS/ops_perf_results_$TS.csv"
echo "[profile] report TS=$TS csv=$CSV"
[ -f "$CSV" ] || { echo "NO CSV FOUND"; ls -la generated/profiler/reports/$TS/; exit 2; }

OUT="perf_reports/${LABEL}_${TS}"
mkdir -p "$OUT"
# phase -> start_signpost end_signpost
declare -A PH=(
  [full]="decode_1_start decode_1_end"
  [prologue]="prologue_start layer_0_start"
  [attn0]="layer_0_start attn_0_end"
  [dense]="mlp_0_start layer_0_end"
  [attn1]="layer_1_start attn_1_end"
  [moe]="moe_start moe_end"
  [lmhead]="lm_head_start lm_head_end"
)
for name in full prologue attn0 dense attn1 moe lmhead; do
  read s e <<< "${PH[$name]}"
  tt-perf-report "$CSV" --start-signpost "$s" --end-signpost "$e" \
    --summary-file "$OUT/$name.summary" > "$OUT/$name.stdout.txt" 2>&1 || echo "  [warn] $name window failed"
done
echo "[profile] DONE -> $OUT"
ls "$OUT"
