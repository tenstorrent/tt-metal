#!/usr/bin/env bash
# Unattended campaign driver: runs rounds, steps, verification and dreaming with no human in the loop.
# Run it detached on the device machine so it survives the orchestrator's terminal going away:
#
#   setsid nohup agent_orch/tools/run_campaign.sh --campaign C --rounds 8 > $DREAM_HOME/C/logs/driver.log 2>&1 &
#
# Round 1 starts at the campaign root; every later round starts at the best valid node so far.
# Touch $DREAM_HOME/<c>/STOP to stop cleanly after the current step. Optional hook: if
# $DREAM_HOME/<c>/publish.sh exists and is executable, it runs after every step (e.g. to publish tree.html).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/dream_env.sh"

CAMPAIGN="" ROUNDS=8 START=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --campaign) CAMPAIGN="$2"; shift 2 ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    --start-round) START="$2"; shift 2 ;;
    *) dream_die "unknown argument $1" ;;
  esac
done
[[ -n "$CAMPAIGN" ]] || dream_die "--campaign is required"
T="$DREAM_TOOLS"
REPO="$(cd "$T/../.." && pwd)"
CHOME="$DREAM_HOME/$CAMPAIGN"
L="$CHOME/ledger"
log() { echo "[driver $(date '+%F %T')] $*"; }
stop_requested() { [[ -f "$CHOME/STOP" ]]; }
ledger_commit() { git -C "$L" add -A && git -C "$L" commit -q --no-verify -m "[dream:$CAMPAIGN] $1" || true; }
publish() { [[ -x "$CHOME/publish.sh" ]] && timeout 600 "$CHOME/publish.sh" >>"$CHOME/logs/publish.log" 2>&1 || true; }
claude_run() {  # claude_run <log> <prompt>
  (cd "$REPO" && timeout 3600 claude -p "$2" --permission-mode bypassPermissions --output-format stream-json \
    --verbose >"$1" 2>&1)
}

best_node() {  # best valid node id over all recorded rounds (ledger overrides applied)
  dream_py - "$CAMPAIGN" <<'PY'
import sys
from dream.campaign import load_campaign
from dream.tree import load_round, recorded_rounds
c = load_campaign(sys.argv[1])
best = None
for r in recorded_rounds(c):
    for n in load_round(c, r).nodes:
        if n.ok and (best is None or n.score > best.score):
            best = n
print(best.node_id if best else "")
PY
}

drift_ok() {  # drift_ok <label>: re-measure the campaign root; fail only on a real shift, not one noisy case
  "$T/eval_attempt.sh" --campaign "$CAMPAIGN" --measure "dream/$CAMPAIGN/root" --label "${1}_root" >>"$CHOME/logs/driver.log" 2>&1
  dream_py - "$L/baseline.json" "$CHOME/reports/${1}_root_1/measure.json" <<'PY2'
import json, math, sys
base, m = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
noise = base["noise_pct"]
d = {k: (v["us_chip_mean"] / base["shapes"][k]["us_chip_mean"] - 1) * 100 for k, v in m["shapes"].items()}
geo = (math.exp(sum(math.log(1 + x / 100) for x in d.values()) / len(d)) - 1) * 100
worst = max(d, key=lambda k: abs(d[k]))
print(f"[driver] root drift: geomean {geo:+.2f}%, worst {worst} {d[worst]:+.2f}% (noise {noise}%)")
sys.exit(0 if m["fail_class"] == "ok" and abs(geo) <= noise and abs(d[worst]) <= 3 * noise else 1)
PY2
}

run_step() {  # run_step <round> -> 0 if a step ran, 1 if the policy stopped
  local t="$1" step="$CHOME/logs/step_r$(printf %02d "$t").json"
  dream_py "$T/policy_step.py" --campaign "$CAMPAIGN" --round "$t" --next >"$step" 2>>"$CHOME/logs/driver_err.log" || {
    log "policy_step failed (see driver_err.log)"; return 1; }
  local items
  items="$(dream_py -c "import json;d=json.load(open('$step'));print('stop' if d['stop'] else ' '.join(i['node']+','+i['parent'] for i in d['items']))")"
  [[ "$items" == "stop" ]] && return 1
  log "round $t step: $items"
  ledger_commit "r$(printf %02d "$t") step"
  local pids=() it node parent
  for it in $items; do
    node="${it%,*}" parent="${it#*,}"
    "$T/prepare_worker.sh" --campaign "$CAMPAIGN" --node "$node" --parent "$parent" >/dev/null || log "prepare $node failed"
    "$T/run_worker.sh" --campaign "$CAMPAIGN" --node "$node" --parent "$parent" >"$CHOME/logs/run_$node.out" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  for it in $items; do
    node="${it%,*}"
    log "$node: $(tail -1 "$CHOME/logs/run_$node.out" | cut -c1-220)"
    dream_py "$T/verify_node.py" --campaign "$CAMPAIGN" --node "$node" --record >>"$CHOME/logs/verify.log" 2>&1 || log "$node: verify issues (recorded)"
  done
  dream_py "$T/history.py" --campaign "$CAMPAIGN" >/dev/null
  ledger_commit "r$(printf %02d "$t") step done"
  publish
  return 0
}

summarize_round() {
  local t="$1" rr; rr="r$(printf %02d "$t")"
  dream_py "$T/history.py" --campaign "$CAMPAIGN" >/dev/null
  mkdir -p "$L/snapshots" && cp "$CHOME/history.md" "$L/snapshots/history_$rr.md"
  claude_run "$CHOME/logs/summary_$rr.jsonl" "Write the round summary for round $t of the Dream-RSI campaign '$CAMPAIGN'.
Read $CHOME/history.md, $L/rounds/$rr/manifest.json and $L/rounds/$rr/decisions.jsonl, and the reflection.md of this
round's nodes (git show dream/$CAMPAIGN/n/<id>:agent_orch/campaigns/$CAMPAIGN/attempts/<id>/reflection.md).
Write $L/rounds/$rr/summary.md with: header (policy, W, R, root and its score), attempt count, best node with per-shape
µs and speedup, a branch x attempt score grid, what worked, what failed and why, policy observations (input for
dreaming: wasted attempts, premature closes, duplicated ideas), and ops notes. Plain factual markdown.
Write only that file. Do not commit. Never push."
  ledger_commit "$rr complete: summary, snapshot"
}

dream_round() {
  local t="$1" cur next rounds
  cur="$(cat "$L/policies/ACTIVE")"
  next="v$(( ${cur#v} + 1 ))"
  rounds="$(seq -s, 1 "$t")"
  dream_py "$T/replay.py" --campaign "$CAMPAIGN" --policy "$L/policies/$cur/policy.py" \
    --out "$L/policies/$cur/replay.json" --traces "$L/policies/$cur/traces.jsonl" >>"$CHOME/logs/driver.log" 2>&1
  claude_run "$CHOME/logs/policy_dev_r$(printf %02d "$t").jsonl" "You are the policy-development agent (dreaming phase) of a Dream-RSI campaign.
Read $REPO/agent_orch/POLICY_DEV.md and follow it exactly.
Inputs: CAMPAIGN=$CAMPAIGN, LEDGER=$L, CURRENT=$cur, ROUNDS=$rounds, M=$(dream_py -c "from dream.campaign import load_campaign as L;print(L('$CAMPAIGN').cfg['dreaming']['revisions'])").
Context: README.md and ORCHESTRATOR.md section 4 in $REPO/agent_orch/, agent_orch/tools/dream/policy_api.py,
$L/rounds/r*/summary.md, $L/rounds/r*/decisions.jsonl, earlier notes in $L/policies/*/notes.md, and $CHOME/history.md.
Rounds after the first start at the previous best node; replay scores them from their root node and stops on
out-of-support batches. Run tools from $REPO (python: $DREAM_PY). Name candidates $cur-cand1..$cur-cand5 and the
winner $next. Write only under $L/policies/. Do not edit ACTIVE, do not commit, never push.
End with only the JSON reply described in POLICY_DEV.md section 6."
  if [[ -f "$L/policies/$next/policy.py" ]]; then
    dream_py "$T/replay.py" --campaign "$CAMPAIGN" --policy "$L/policies/$next/policy.py" \
      --out "$L/policies/$next/replay.json" --traces "$L/policies/$next/traces.jsonl" >>"$CHOME/logs/driver.log" 2>&1
    local better
    better="$(dream_py -c "import json;a=json.load(open('$L/policies/$cur/replay.json'))['objective_V'];b=json.load(open('$L/policies/$next/replay.json'))['objective_V'];print(int(b>=a))")"
    if [[ "$better" == 1 ]]; then echo "$next" >|"$L/policies/ACTIVE"; log "activated $next"; else log "$next did not beat $cur; keeping $cur"; fi
  else
    log "dreaming produced no $next; keeping $cur"
  fi
  find "$L/policies" -name __pycache__ -prune -exec rm -rf {} +
  ledger_commit "dreaming after r$(printf %02d "$t"): active $(cat "$L/policies/ACTIVE")"
}

[[ -f "$L/baseline.json" ]] || dream_die "no baseline; run eval_attempt.sh --baseline first"
start="${START:-1}"
log "campaign $CAMPAIGN: rounds $start..$ROUNDS"
for t in $(seq "$start" "$ROUNDS"); do
  stop_requested && { log "STOP file found"; break; }
  rr="r$(printf %02d "$t")"
  if [[ ! -f "$L/rounds/$rr/manifest.json" ]]; then
    root_ref="dream/$CAMPAIGN/root"
    if [[ $t -gt 1 ]]; then b="$(best_node)"; [[ -n "$b" ]] && root_ref="dream/$CAMPAIGN/n/$b"; fi
    if ! drift_ok "$rr"; then
      log "re-measuring the root once more before deciding"
      drift_ok "${rr}b" || { log "baseline drift confirmed before $rr, stopping (see $CHOME/reports/${rr}*_root_1)"; break; }
    fi
    dream_py "$T/history.py" --campaign "$CAMPAIGN" >/dev/null
    dream_py "$T/policy_step.py" --campaign "$CAMPAIGN" --round "$t" --plan --root "$root_ref" >>"$CHOME/logs/driver.log" 2>&1 || {
      log "plan failed"; break; }
    ledger_commit "$rr plan (root $root_ref)"
    log "$rr planned, root $root_ref, policy $(cat "$L/policies/ACTIVE")"
  fi
  while run_step "$t"; do
    stop_requested && break
  done
  stop_requested && { log "STOP file found after a step of $rr"; break; }
  summarize_round "$t"
  log "$rr complete; best so far: $(best_node)"
  publish
  [[ $t -lt $ROUNDS ]] && dream_round "$t"
done
dream_py "$T/history.py" --campaign "$CAMPAIGN" >/dev/null
publish
log "driver finished"
