#!/usr/bin/env bash
# Prepare the worktree for one worker and write the skeleton node.json.
#
#   prepare_worker.sh --campaign C --node r01-b03-a01 --parent root
#   prepare_worker.sh --campaign C --node r01-b03-a02 --parent r01-b03-a01
#
# a01 creates branch dream/<c>/b/rTT-bBB + its worktree at the round root (from the round
# manifest); later attempts reuse the worktree and check that it sits exactly on the parent.
# Prints the worktree path on the last line.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/dream_env.sh"

CAMPAIGN="" NODE="" PARENT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --campaign) CAMPAIGN="$2"; shift 2 ;;
    --node) NODE="$2"; shift 2 ;;
    --parent) PARENT="$2"; shift 2 ;;
    *) dream_die "unknown argument $1" ;;
  esac
done
[[ -n "$CAMPAIGN" && -n "$NODE" && -n "$PARENT" ]] || dream_die "--campaign, --node and --parent are required"
[[ "$NODE" =~ ^r([0-9]{2})-b([0-9]{2})-a([0-9]{2})$ ]] || dream_die "bad node id $NODE"
RR="${BASH_REMATCH[1]}" BB="${BASH_REMATCH[2]}" AA="${BASH_REMATCH[3]}"
CHOME="$DREAM_HOME/$CAMPAIGN"
WT="$CHOME/wt/r$RR-b$BB"
BRANCH="dream/$CAMPAIGN/b/r$RR-b$BB"
MANIFEST="$CHOME/ledger/rounds/r$RR/manifest.json"
[[ -f "$MANIFEST" ]] || dream_die "no manifest for round $RR (run policy_step.py --plan first)"
git rev-parse -q --verify "refs/tags/dream/$CAMPAIGN/n/$NODE" >/dev/null && dream_die "$NODE already exists"

if [[ "$AA" == "01" ]]; then
  [[ "$PARENT" == "root" ]] || dream_die "a01 must start from root"
  round_root="$(dream_py -c "import json;print(json.load(open('$MANIFEST'))['round_root_commit'])")"
  parent_commit="$round_root"
  if [[ -d "$WT" ]]; then  # a previous a01 worker died before committing
    [[ "$(git -C "$WT" rev-parse HEAD)" == "$round_root" ]] || dream_die "$WT exists and is not at the round root"
    git -C "$WT" reset -q --hard && git -C "$WT" clean -qfd
  else
    git worktree add -q -b "$BRANCH" "$WT" "$round_root"
  fi
else
  [[ -d "$WT" ]] || dream_die "no worktree $WT for branch $BRANCH"
  parent_commit="$(git rev-parse "dream/$CAMPAIGN/n/$PARENT^{commit}")" || dream_die "parent $PARENT has no tag"
  [[ "$(git -C "$WT" rev-parse HEAD)" == "$parent_commit" ]] || dream_die "$WT HEAD is not $PARENT"
  git -C "$WT" reset -q --hard && git -C "$WT" clean -qfd  # leftovers from a lost attempt
fi

NODE_DIR="$WT/agent_orch/campaigns/$CAMPAIGN/attempts/$NODE"
mkdir -p "$NODE_DIR"
dream_py - "$NODE_DIR/node.json" <<PY
import json, sys, datetime
json.dump({
  "node_id": "$NODE", "campaign": "$CAMPAIGN",
  "round": int("$RR"), "branch": int("$BB"), "attempt": int("$AA"),
  "parent": "$PARENT", "parent_commit": "$parent_commit",
  "worker": "", "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
  "mechanism": "", "tags": [],
}, open(sys.argv[1], "w"), indent=2)
PY
echo "$WT"
