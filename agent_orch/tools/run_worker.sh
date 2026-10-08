#!/usr/bin/env bash
# Launch one worker as a headless Claude Code session inside its worktree (on the device machine).
#
#   run_worker.sh --campaign C --node r01-b02-a01 --parent root [--timeout-min 120] [--model M]
#   run_worker.sh --campaign C --node N --parent P --attach <pid>          # re-watch a running session
#   run_worker.sh --campaign C --node N --parent P --resume <session-id>   # continue an interrupted session
#
# Run prepare_worker.sh first. The transcript goes to $DREAM_HOME/<c>/logs/worker_<node>.jsonl;
# the worker's final message (the commit_node.py JSON) is printed on the last line.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/dream_env.sh"

CAMPAIGN="" NODE="" PARENT="" TIMEOUT_MIN=120 MODEL="" ATTACH="" RESUME=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --campaign) CAMPAIGN="$2"; shift 2 ;;
    --node) NODE="$2"; shift 2 ;;
    --parent) PARENT="$2"; shift 2 ;;
    --timeout-min) TIMEOUT_MIN="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --attach) ATTACH="$2"; shift 2 ;;
    --resume) RESUME="$2"; shift 2 ;;
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

TAG="dream/$CAMPAIGN/n/$NODE"
cd "$WT" || exit 1
if [[ -n "$ATTACH" ]]; then
  pid="$ATTACH"
  echo "[worker $NODE] attached to pid $pid $(date -Is), log: $LOG"
else
  args=(-p "$PROMPT")
  if [[ -n "$RESUME" ]]; then
    args=(-p --resume "$RESUME" "Your session was interrupted. Check what is already done in your node directory
(eval/ may already be written), finish the remaining WORKER.md steps (reflection.md, commit_node.py) without
re-running a finished eval, and end with only the JSON line printed by commit_node.py.")
  fi
  echo "[worker $NODE] started $(date -Is), log: $LOG"
  timeout --kill-after=60 "$((TIMEOUT_MIN * 60))" \
    claude "${args[@]}" --permission-mode bypassPermissions --output-format stream-json --verbose \
    ${MODEL:+--model "$MODEL"} >>"$LOG" 2>&1 &
  pid=$!
fi
# The session ends on its own when the worker is done. A "result" event is NOT the end: the
# worker emits one each time it waits on a background task (e.g. its eval). The only reliable
# "done" signal is the node tag. A worker can leave a background shell running after it has
# committed, which keeps the session alive; in that case wait 2 minutes, then clean up.
while kill -0 "$pid" 2>/dev/null; do
  if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
    for _ in $(seq 1 24); do kill -0 "$pid" 2>/dev/null || break; sleep 5; done
    if kill -0 "$pid" 2>/dev/null; then
      echo "[worker $NODE] committed but the session is still running; stopping leftover processes"
      kill_tree "$pid"
    fi
    break
  fi
  sleep 10
done
wait "$pid" 2>/dev/null
rc=$?
echo "[worker $NODE] exited rc=$rc $(date -Is), tag $(git rev-parse -q --verify "refs/tags/$TAG" >/dev/null && echo present || echo MISSING)"
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
