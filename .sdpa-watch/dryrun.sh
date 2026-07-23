#!/usr/bin/env bash
# Dry-run the SDPA watcher agent against ONE specific run, printing the digest
# block instead of posting. Reuses the real prompt template + real config so
# it matches what watch.sh would produce.
#
#   Usage: dryrun.sh <workflow.yaml> <run_id>
#   e.g.   dryrun.sh t3000-e2e-tests.yaml 29898728562
set -euo pipefail

SDPA_HOME="$HOME/.sdpa-watch"
source "$SDPA_HOME/config.sh"          # REPO, MODEL, TT_METAL_DIR, PATH self-heal, PIPELINES
PROMPT_TEMPLATE="$SDPA_HOME/agent_prompt.txt"

WF="$1"; RID="$2"

# Pull this pipeline's real entry (display|hint|job_pattern) from config.sh.
entry=""
for p in "${PIPELINES[@]}"; do [[ "$p" == "$WF|"* ]] && entry="$p" && break; done
[[ -z "$entry" ]] && { echo "no PIPELINES entry for $WF" >&2; exit 1; }
IFS='|' read -r workflow display test_hint job_pattern <<<"$entry"

# --- fetch job-filtered failure logs (same extraction as watch.sh) ----------
failed_jobs=$(gh api "repos/$REPO/actions/runs/$RID/jobs" --paginate \
              --jq '.jobs[] | select(.conclusion=="failure") | "\(.id)\t\(.name)"')
[[ -n "$job_pattern" ]] && failed_jobs=$(printf '%s\n' "$failed_jobs" | grep -E -i "$job_pattern" || true)
echo ">>> in-scope failed jobs:" >&2; printf '%s\n' "$failed_jobs" | cut -f2 >&2

logs=""
while IFS=$'\t' read -r jid jname; do
  [[ -z "$jid" ]] && continue
  full=$(gh api "repos/$REPO/actions/jobs/$jid/logs" 2>/dev/null)
  cut=$(printf '%s' "$full" | { grep -E -B 5 -A 100 '##\[error\]|FAILED |AssertionError|Traceback' || true; })
  [[ -z "$cut" ]] && cut=$(printf '%s' "$full" | tail -c 12000)
  logs+="=== JOB: $jname ===
$cut

"
done <<<"$failed_jobs"

# --- build the identical prompt and run the agent ---------------------------
read -r rsha rurl < <(gh api "repos/$REPO/actions/runs/$RID" --jq '"\(.head_sha) \(.html_url)"')
ctx="Pipeline display name: $display
Workflow file: $workflow
Run: #$RID  conclusion=failure  sha=$rsha
URL: $rurl
Test focus hint: $test_hint

Commits since last analyzed run: (dry run — not tracked)

Failure log excerpt (truncated to last ~40k chars):
$logs"

prompt="$(cat "$PROMPT_TEMPLATE")

# Context
$ctx"

echo ">>> asking $MODEL ..." >&2
echo "================================================================"
cd "$TT_METAL_DIR" && claude --model "$MODEL" -p <<<"$prompt" | sed -n '/^▸/,$p'
echo "================================================================"
