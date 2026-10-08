#!/usr/bin/env bash
# One-time campaign setup. Run from a checkout of the repo whose HEAD (or --base) contains
# agent_orch/campaigns/<c>/campaign.yaml.
#
#   setup_campaign.sh --campaign rmsnorm-prefill [--base <ref>] [--no-build]
#
# Creates: tag dream/<c>/root, $DREAM_HOME/<c>/{wt,reports,logs}, the ledger branch + worktree
# (with policies/v0 and ACTIVE=v0), and the eval checkout (built unless --no-build).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/dream_env.sh"

CAMPAIGN="" BASE="HEAD" BUILD=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --campaign) CAMPAIGN="$2"; shift 2 ;;
    --base) BASE="$2"; shift 2 ;;
    --no-build) BUILD=0; shift ;;
    *) dream_die "unknown argument $1" ;;
  esac
done
[[ -n "$CAMPAIGN" ]] || dream_die "--campaign is required"
REPO="$(git rev-parse --show-toplevel)"
CHOME="$DREAM_HOME/$CAMPAIGN"
ROOT="dream/$CAMPAIGN/root"
base_sha="$(git rev-parse "$BASE^{commit}")"
git cat-file -e "$base_sha:agent_orch/campaigns/$CAMPAIGN/campaign.yaml" 2>/dev/null ||
  dream_die "$BASE does not contain agent_orch/campaigns/$CAMPAIGN/campaign.yaml (commit it first)"

# root tag
if git rev-parse -q --verify "refs/tags/$ROOT" >/dev/null; then
  [[ "$(git rev-parse "$ROOT^{commit}")" == "$base_sha" ]] || dream_die "$ROOT already exists at a different commit"
else
  git tag "$ROOT" "$base_sha"
  echo "tagged $ROOT = ${base_sha:0:12}"
fi
mkdir -p "$CHOME"/{wt,reports,logs} "$CCACHE_DIR"

# ledger: an orphan branch holding policies, baseline, round manifests and decisions
LEDGER="$CHOME/ledger"
if ! git rev-parse -q --verify "refs/heads/dream/$CAMPAIGN/ledger" >/dev/null; then
  empty_tree="$(git hash-object -t tree /dev/null)"
  init="$(git commit-tree "$empty_tree" -m "[dream:$CAMPAIGN] ledger init")"
  git branch "dream/$CAMPAIGN/ledger" "$init"
fi
if [[ ! -d "$LEDGER" ]]; then
  git worktree add -q "$LEDGER" "dream/$CAMPAIGN/ledger"
fi
if [[ ! -f "$LEDGER/policies/ACTIVE" ]]; then
  mkdir -p "$LEDGER/policies/v0" "$LEDGER/rounds"
  git show "$base_sha:agent_orch/policies/v0/policy.py" >"$LEDGER/policies/v0/policy.py"
  printf '# v0: parallel refine\n\nHand-written baseline policy (paper: fixed exploration). See policy.py docstring.\n' \
    >"$LEDGER/policies/v0/notes.md"
  echo v0 >"$LEDGER/policies/ACTIVE"
  git -C "$LEDGER" add -A
  git -C "$LEDGER" commit -q --no-verify -m "[dream:$CAMPAIGN] ledger: add policy v0"
  echo "ledger ready at $LEDGER (ACTIVE=v0)"
fi

# eval checkout: the only place attempts are built and run
EV="$CHOME/eval"
if [[ ! -d "$EV" ]]; then
  git worktree add -q --detach "$EV" "$ROOT"
  git -C "$EV" submodule update --init --recursive
  echo "eval checkout at $EV"
fi
if [[ $BUILD == 1 && ! -f "$EV/.dream_last_build" ]]; then
  echo "building eval checkout (first build is slow; log: $CHOME/logs/build_setup.log)"
  (cd "$EV" && ./build_metal.sh --release --enable-ccache --cpm-source-cache "$DREAM_CPM_CACHE") \
    >"$CHOME/logs/build_setup.log" 2>&1 || dream_die "build failed, see $CHOME/logs/build_setup.log"
  git -C "$EV" rev-parse HEAD >"$EV/.dream_last_build"
fi
echo "done. next: eval_attempt.sh --campaign $CAMPAIGN --baseline, then commit $LEDGER/baseline.json"
