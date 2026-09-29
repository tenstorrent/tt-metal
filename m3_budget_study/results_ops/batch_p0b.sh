#!/bin/bash
# P0-B (a): per-chip MoE load of real routing on the (2,4) stage-0 carve, layers 0-6, no profiler.
#
# One run_budget.sh process per case (reset, lock check, watchdog), M3_MOE_LOAD_STATS=1 with the raw counts in
# skew/<id>.jsonl, then tools/load_skew.py -> load_skew.csv (one row per timed point x MoE layer) and a row in
# runs.csv. Same env as the P0-A zone profiles: M3_MOE_W_NDSHARD=1 M3_MOE_HYBRID_THRESHOLD=128, v1
# dispatch/combine, 1d fabric. Tokens: longbook_56320 (prose) and inputs/code_m3 (code). h=141312 is not a
# multiple of W=4096; the single cases use 139264, the depth P0-A's harness rounds 141312 down to.
#
#   nohup m3_budget_study/results_ops/batch_p0b.sh > m3_budget_study/results_ops/logs/batch_p0b.out 2>&1 &
#   ONLY="skew_single_prose" ...   run a subset
set -uo pipefail
RES="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$RES/../.." && pwd)}"
LOCK="$RES/.lock"
GOLDEN=/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill/golden
PROSE="$GOLDEN/longbook_56320/metadata.json"
CODE="$RES/inputs/code_m3/metadata.json"
PY="$TT_METAL_HOME/python_env/bin/python"
mkdir -p "$RES/skew" "$RES/logs"

export BUDGET_RESULTS="$RES" BUDGET_LOCK="$LOCK" BUDGET_LOCK_OWNER=vmelnykov-ops-agent
export BUDGET_COLLECT_CSV="$RES/skew/budget_runs.csv" BUDGET_STAGES=4 BUDGET_STAGE=0 BUDGET_LAYER_IDS=0,1,2,3,4,5,6
export M3_FABRIC=1d M3_MOE_W_NDSHARD=1 M3_MOE_HYBRID_THRESHOLD=128 M3_MOE_DISPATCH=v1 M3_MOE_COMBINE=v1
export EXPERT_DTYPE=bf4 M3_MOE_LOAD_STATS=1 BUDGET_WARMUP=1 BUDGET_ITERS=1
export LOAD_TIMEOUT=900 STALL_TIMEOUT=300 RUN_TIMEOUT="${RUN_TIMEOUT:-1200}"

# id | harness | W | B | load_skew --input | extra env (space-separated K=V)
CASES=(
  "skew_single_prose|budget_sweep.py|4096|1|prose|BUDGET_W=4096 BUDGET_POINTS=0:4096,139264:4096 BUDGET_TOKENS=$PROSE"
  "skew_single_code|budget_sweep.py|4096|1|code|BUDGET_W=4096 BUDGET_POINTS=0:4096,139264:4096 BUDGET_TOKENS=$CODE"
  "skew_packed_w4096|budget_packed.py|4096|2|prose+code|BUDGET_B=2 BUDGET_TOKENS=$PROSE BUDGET_INPUTS=prose=$PROSE;code=$CODE BUDGET_COMPOS=mixed=prose@141312:2048,code@0:2048"
  "skew_packed_w8192|budget_packed.py|8192|4|prose+code+prose+code|BUDGET_B=4 BUDGET_TOKENS=$PROSE BUDGET_INPUTS=prose=$PROSE;code=$CODE BUDGET_COMPOS=mixed=prose@548864:2048,code@141312:2048,prose.1@16384:2048,code.1@0:2048"
)

for spec in "${CASES[@]}"; do
  IFS='|' read -r id harness W B input extra <<< "$spec"
  if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $id "* ]]; then continue; fi
  log="$RES/logs/$id.log"
  if [ -e "$log" ]; then echo "[p0b] $id: $log exists, skipping"; continue; fi
  busy="$(pgrep -u "$(id -u)" -af 'models/demos/minimax_m3/tests/perf/[a-z_]+\.py|tracy-capture' || true)"
  if [ -n "$busy" ]; then echo "[p0b] another harness is running, stopping before $id:"$'\n'"$busy"; exit 1; fi
  stats="$RES/skew/$id.jsonl"; rm -f "$stats"
  kv=(); IFS=' ' read -r -a kv <<< "$extra"
  start=$(date -u +%FT%TZ)
  echo "[p0b] $start START $id"
  env "${kv[@]}" RUN_ID="$id" HARNESS="$harness" M3_MOE_LOAD_STATS_FILE="$stats" "$TT_METAL_HOME/m3_budget_study/run_budget.sh"
  end=$(date -u +%FT%TZ)
  status="$(grep -oE '^STATUS=[A-Z_]+' "$log" | tail -1 | cut -d= -f2)"
  echo "[p0b] $end END $id status=${status:-?}"
  notes="load stats -> skew/$id.jsonl"
  if [ "$status" = OK ]; then
    "$PY" "$RES/tools/load_skew.py" --stats "$stats" --log "$log" --case "$id" --W "$W" --B "$B" --input "$input" \
      --out "$RES/load_skew.csv" > "$RES/skew/$id.skew.txt" 2>&1 || notes="$notes; load_skew.py FAILED"
    tail -1 "$RES/skew/$id.skew.txt"
  fi
  envline="$(grep -vE '^(run_id|git_sha|dirty|date)=' "$RES/logs/$id.env" | sort | tr '\n' ' ' | sed 's/ $//')"
  sha="$(grep '^git_sha=' "$RES/logs/$id.env" | cut -d= -f2)"
  dirty="$(git -C "$TT_METAL_HOME" status --porcelain -uno | awk '{print $2}' | tr '\n' ' ' | sed 's/ $//')"
  [ -n "$dirty" ] && sha="$sha dirty($(echo "$dirty" | wc -w): $dirty)"
  "$PY" - "$RES/runs.csv" "$id" "$start" "$end" "$sha" "$envline" \
    "HARNESS=$harness RUN_ID=$id $extra m3_budget_study/run_budget.sh" "STATUS=${status:-?}" \
    "m3_budget_study/results_ops/logs/$id.log" "P0B $notes" <<'PY'
import csv, sys
out, rid, start, end, sha, env, cmd, status, log, notes = sys.argv[1:]
with open(out, "a", newline="") as f:
    csv.writer(f).writerow([rid, "P0B", start, end, sha, env, cmd, status, log, notes])
PY
done
echo "[p0b] $(date -u +%FT%TZ) done"
