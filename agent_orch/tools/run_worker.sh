#!/usr/bin/env bash
# Launch one worker as a headless Claude Code session inside its worktree (on the device machine).
#
#   run_worker.sh --campaign C --node r01-b02-a01 --parent root [--timeout-min 120] [--model M]
#
# Run prepare_worker.sh first. The transcript goes to $DREAM_HOME/<c>/logs/worker_<node>.jsonl;
# the worker's final message (the commit_node.py JSON) is printed on the last line.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/dream_env.sh"

CAMPAIGN="" NODE="" PARENT="" TIMEOUT_MIN=120 MODEL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --campaign) CAMPAIGN="$2"; shift 2 ;;
    --node) NODE="$2"; shift 2 ;;
    --parent) PARENT="$2"; shift 2 ;;
    --timeout-min) TIMEOUT_MIN="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    *) dream_die "unknown argument $1" ;;
  esac
done
[[ "$NODE" =~ ^r([0-9]{2})-b([0-9]{2})-a[0-9]{2}$ ]] || dream_die "bad node id $NODE"
CHOME="$DREAM_HOME/$CAMPAIGN"
WT="$CHOME/wt/r${BASH_REMATCH[1]}-b${BASH_REMATCH[2]}"
[[ -f "$WT/agent_orch/campaigns/$CAMPAIGN/attempts/$NODE/node.json" ]] || dream_die "run prepare_worker.sh first"
LOG="$CHOME/logs/worker_$NODE.jsonl"
mkdir -p "$CHOME/logs"

PROMPT="You are a worker in a Dream-RSI discovery campaign that optimizes a tt-metal op on Tenstorrent Blackhole hardware.
Read $WT/agent_orch/WORKER.md and follow it exactly. It defines your whole job.

Inputs:
- CAMPAIGN=$CAMPAIGN
- NODE_ID=$NODE
- PARENT=$PARENT
- WORKTREE=$WT
- HISTORY=$CHOME/history.md
- DREAM_HOME=$DREAM_HOME (already the default in the tools)

Work only inside WORKTREE (your current directory). Run the tools from it:
agent_orch/tools/eval_attempt.sh and agent_orch/tools/commit_node.py.
Set \"worker\" in node.json to your model id.
Never push, never edit files outside WORKTREE, never kill processes you did not start, never reset the device.
Your final message must be only the JSON line printed by commit_node.py."

kill_tree() {  # kill a process and all its descendants
  local p
  for p in $(pgrep -P "$1"); do kill_tree "$p"; done
  kill "$1" 2>/dev/null
}

echo "[worker $NODE] started $(date -Is), log: $LOG"
cd "$WT" || exit 1
timeout --kill-after=60 "$((TIMEOUT_MIN * 60))" \
  claude -p "$PROMPT" --permission-mode bypassPermissions --output-format stream-json --verbose \
  ${MODEL:+--model "$MODEL"} >"$LOG" 2>&1 &
pid=$!
# A worker can leave a background shell running after its final message, which keeps the
# session alive. Once the result is in the log, give it a minute to exit, then clean up.
while kill -0 "$pid" 2>/dev/null; do
  if grep -q '"type":"result"' "$LOG" 2>/dev/null; then
    for _ in $(seq 1 12); do kill -0 "$pid" 2>/dev/null || break; sleep 5; done
    if kill -0 "$pid" 2>/dev/null; then
      echo "[worker $NODE] result received but the session is still running; stopping leftover processes"
      kill_tree "$pid"
    fi
    break
  fi
  sleep 10
done
wait "$pid"
rc=$?
echo "[worker $NODE] exited rc=$rc $(date -Is)"
"$DREAM_PY" - "$LOG" <<'PY'
import json, sys
res = None
for line in open(sys.argv[1], errors="replace"):
    try:
        d = json.loads(line)
    except ValueError:
        continue
    if d.get("type") == "result":
        res = d
if res is None:
    print(json.dumps({"error": "no result in worker log"}))
else:
    print(f"[cost ${res.get('total_cost_usd', 0):.2f}, turns {res.get('num_turns')}, {res.get('duration_ms', 0) // 60000} min]")
    print((res.get("result") or "").strip().splitlines()[-1] if res.get("result") else json.dumps({"error": "empty result"}))
PY
