#!/bin/bash
# One budget_sweep.py process with reset, watchdog and per-run log/env capture.
# Usage: RUN_ID=e1_s8_w2048 EXP=E1 LAYER_SET=S8 BUDGET_LAYER_IDS=8,...,15 BUDGET_W=2048 BUDGET_POINTS=0:2048 ./run_budget.sh
# Watchdog: LOAD_TIMEOUT s until the model is built, then STALL_TIMEOUT s without log output -> kill.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$HERE/.." && pwd)}"
RES="$HERE/results"
: "${RUN_ID:?}" "${BUDGET_LAYER_IDS:?}"; [ -n "${HARNESS:-}" ] && export HARNESS
export HF_MODEL="${HF_MODEL:-/mnt/weka/model-weights/llm/minimax/MiniMax-M3}"
export TT_CACHE_PATH="${TT_CACHE_PATH:-/mnt/weka/model-cache/scratch/minimax/MiniMax-M3-cache/prefill}"
export BUDGET_TOKENS="${BUDGET_TOKENS:-$TT_CACHE_PATH/golden/longbook_qa_eng_prefill_56320_nopad/metadata.json}"
export M3_FABRIC="${M3_FABRIC:-1d}" EXPERT_DTYPE="${EXPERT_DTYPE:-bf4}" LOGURU_LEVEL=INFO
export TT_MESH_GRAPH_DESC_PATH="${TT_MESH_GRAPH_DESC_PATH:-$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto}"
LOAD_TIMEOUT="${LOAD_TIMEOUT:-1500}" STALL_TIMEOUT="${STALL_TIMEOUT:-300}"
LOG="$RES/logs/$RUN_ID.log"; ENVF="$RES/logs/$RUN_ID.env"
[ -e "$LOG" ] && { echo "refusing to overwrite $LOG"; exit 2; }
cd "$TT_METAL_HOME"; source python_env/bin/activate; export PYTHONPATH="$TT_METAL_HOME"
ulimit -Su "$(ulimit -Hu)" 2>/dev/null
{ echo "run_id=$RUN_ID"; echo "git_sha=$(git rev-parse HEAD)"; echo "dirty=$(git status --porcelain -uno | wc -l)"
  echo "date=$(date -Is)"; env | grep -E '^(BUDGET_|M3_|TT_|EXPERT_|HF_|EXP=|LAYER_SET=)' | sort; } > "$ENVF"
tt-smi -glx_reset > "$RES/logs/$RUN_ID.reset" 2>&1 || { echo "STATUS=ERROR reset failed" | tee -a "$LOG"; exit 1; }
python3 -u models/demos/minimax_m3/tests/perf/${HARNESS:-budget_sweep.py} > "$LOG" 2>&1 &
PID=$!; T0=$(date +%s); STATUS=""
while kill -0 $PID 2>/dev/null; do
  sleep 10; now=$(date +%s); age=$(( now - $(stat -c %Y "$LOG") ))
  if ! grep -q '"kind": "built"' "$LOG"; then
    [ $((now - T0)) -gt "$LOAD_TIMEOUT" ] && { STATUS=LOAD_TIMEOUT; break; }
  elif [ "$age" -gt "$STALL_TIMEOUT" ]; then STATUS=HANG; break; fi
done
if [ -n "$STATUS" ]; then kill -TERM $PID; sleep 15; kill -KILL $PID 2>/dev/null; pkill -KILL -P $PID 2>/dev/null
else wait $PID; rc=$?
  if grep -q '"kind": "done"' "$LOG"; then STATUS=OK
  elif grep -qiE 'out of memory|OOM|Out of Memory' "$LOG"; then STATUS=OOM; else STATUS=ERROR; fi
fi
echo "STATUS=$STATUS elapsed=$(( $(date +%s) - T0 ))s" | tee -a "$LOG"
[ "$STATUS" = OK ] || tt-smi -glx_reset >> "$RES/logs/$RUN_ID.reset" 2>&1
python3 "$HERE/budget_collect.py" "$LOG" "$ENVF" "$RES/runs.csv"
