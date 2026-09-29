#!/bin/bash
# P1-E: does dense attention scan the whole KV slot? Dense layers at h=16384, W=4096, slot capacity 64k vs 1M.
# Pavlo's fit: ~100 ms per 1M-token slot per dense layer per chunk (whole-capacity scan); our E2c on (4,4): +2%.
#
#  A. budget_sweep.py on the (2,4) stage-0 carve, layers 0-2 (dense, contiguous from 0), BUDGET_CAPACITY
#     65536 vs 1048576, points 0:4096 (cold control) and 16384:4096, real-token fill, BUDGET_MEM=1
#     (DRAM in use after the last point). One run_budget.sh process each (reset, lock, watchdog, runs.csv row).
#  B. the common runner, 4 x (2,4) with one layer per stage (PREFILL_NUM_LAYERS=4, PREFILL_PP_LAYER_COUNTS=1,1,1,1:
#     stages 0-2 are the dense layers 0-2, stage 3 is sparse layer 3), PREFILL_MAX_SEQ_LEN 65536 vs 1048576,
#     2 users, sync per chunk: per-stage compute of one dense layer, cold and at h=16384 (synthetic prefix).
#     Runs through batch_p1d.sh (P1D_SESSIONS), results in pipeline/p1e_*.
# Fixed env: M3_MOE_W_NDSHARD=1 M3_MOE_HYBRID_THRESHOLD=128, v1 dispatch/combine; 1d fabric for A,
# PREFILL_FABRIC_MODE=2d (4-rank manifest) for B. Tokens: longbook_56320.
# Readout: tools/dense_scan_check.py -> dense_scan_check.txt.
#
#   nohup m3_budget_study/results_ops/batch_p1e.sh > m3_budget_study/results_ops/logs/batch_p1e.out 2>&1 &
#   ONLY="p1e_bs_cap65536 p1e_rn_cap64k" ...   run a subset;   --dry-run / DRY_RUN=1: print the plan only
set -uo pipefail
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1
DRY_RUN="${DRY_RUN:-0}"
RES="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$RES/../.." && pwd)}"
LOCK="$RES/.lock"
PROSE=/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill/golden/longbook_56320/metadata.json
PY="$TT_METAL_HOME/python_env/bin/python"
mkdir -p "$RES/dense_scan" "$RES/logs"

export BUDGET_RESULTS="$RES" BUDGET_LOCK="$LOCK" BUDGET_LOCK_OWNER=vmelnykov-ops-agent
export BUDGET_COLLECT_CSV="$RES/dense_scan/budget_runs.csv" BUDGET_STAGES=4 BUDGET_STAGE=0 BUDGET_LAYER_IDS=0,1,2
export M3_FABRIC=1d M3_MOE_W_NDSHARD=1 M3_MOE_HYBRID_THRESHOLD=128 M3_MOE_DISPATCH=v1 M3_MOE_COMBINE=v1
export EXPERT_DTYPE=bf4 BUDGET_W=4096 BUDGET_POINTS=0:4096,16384:4096 BUDGET_WARMUP=3 BUDGET_ITERS=8 BUDGET_MEM=1
export BUDGET_TOKENS="$PROSE" LOAD_TIMEOUT=900 STALL_TIMEOUT=300 RUN_TIMEOUT="${RUN_TIMEOUT:-1200}"

in_only () { [ -z "${ONLY:-}" ] || [[ " $ONLY " == *" $1 "* ]]; }

for cap in 65536 1048576; do
  id="p1e_bs_cap$cap"
  in_only "$id" || continue
  if [ "$DRY_RUN" = 1 ]; then
    echo "[p1e] DRY $id: RUN_ID=$id HARNESS=budget_sweep.py BUDGET_CAPACITY=$cap BUDGET_LAYER_IDS=$BUDGET_LAYER_IDS" \
         "BUDGET_W=$BUDGET_W BUDGET_POINTS=$BUDGET_POINTS BUDGET_MEM=1 m3_budget_study/run_budget.sh"
    continue
  fi
  log="$RES/logs/$id.log"
  if [ -e "$log" ]; then echo "[p1e] $id: $log exists, skipping"; continue; fi
  busy="$(pgrep -u "$(id -u)" -af 'models/demos/minimax_m3/tests/perf/[a-z_]+\.py|tracy-capture|prefill\.runners\.prefill_(runner|producer)' || true)"
  if [ -n "$busy" ]; then echo "[p1e] another harness is running, stopping before $id:"$'\n'"$busy"; exit 1; fi
  start=$(date -u +%FT%TZ); echo "[p1e] $start START $id"
  RUN_ID="$id" HARNESS=budget_sweep.py BUDGET_CAPACITY=$cap "$TT_METAL_HOME/m3_budget_study/run_budget.sh"
  end=$(date -u +%FT%TZ)
  status="$(grep -oE '^STATUS=[A-Z_]+' "$log" | tail -1 | cut -d= -f2)"
  echo "[p1e] $end END $id status=${status:-?}"
  envline="$(grep -vE '^(run_id|git_sha|dirty|date)=' "$RES/logs/$id.env" | sort | tr '\n' ' ' | sed 's/ $//')"
  sha="$(grep '^git_sha=' "$RES/logs/$id.env" | cut -d= -f2)"
  dirty="$(git -C "$TT_METAL_HOME" status --porcelain -uno | awk '{print $2}' | tr '\n' ' ' | sed 's/ $//')"
  [ -n "$dirty" ] && sha="$sha dirty($(echo "$dirty" | wc -w): $dirty)"
  "$PY" - "$RES/runs.csv" "$id" "$start" "$end" "$sha" "$envline" \
    "HARNESS=budget_sweep.py RUN_ID=$id BUDGET_CAPACITY=$cap m3_budget_study/run_budget.sh" "STATUS=${status:-?}" \
    "m3_budget_study/results_ops/logs/$id.log" "P1E dense layers 0-2, capacity $cap" <<'PY'
import csv, sys
out, rid, start, end, sha, env, cmd, status, log, notes = sys.argv[1:]
with open(out, "a", newline="") as f:
    csv.writer(f).writerow([rid, "P1E", start, end, sha, env, cmd, status, log, notes])
PY
done

# B: runner, one layer per stage
RUNNER_SESSIONS=""
for spec in "p1e_rn_cap64k|65536" "p1e_rn_cap1m|1048576"; do
  id="${spec%%|*}" cap="${spec#*|}"
  in_only "$id" || continue
  RUNNER_SESSIONS+="$id|4096|1|1|h16k_streams|PREFILL_NUM_LAYERS=4 PREFILL_PP_LAYER_COUNTS=1,1,1,1 PREFILL_MAX_SEQ_LEN=$cap"$'\n'
done
if [ -n "$RUNNER_SESSIONS" ]; then
  P1D_SESSIONS="${RUNNER_SESSIONS%$'\n'}" DRY_RUN="$DRY_RUN" RUN_TIMEOUT=900 "$RES/batch_p1d.sh"
fi
[ "$DRY_RUN" = 1 ] && exit 0
"$PY" "$RES/tools/dense_scan_check.py" --out "$RES/dense_scan_check.txt" || echo "[p1e] readout failed"
echo "[p1e] $(date -u +%FT%TZ) done"
