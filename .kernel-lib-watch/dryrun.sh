#!/usr/bin/env bash
# Dry-run the kernel-lib watcher agent against ONE specific run, printing the
# digest block instead of posting. Reuses the real prompt template + real config
# so it matches what watch.sh would produce.
#
# Self-locating (like this dir's watch.sh): it reads the config/prompt sitting
# next to THIS file, so it never picks up ~/.sdpa-watch or ~/.conv-watch by
# accident.
#
# Scheduled configurations are selected automatically from the run's event and
# title. Manual workflow_dispatch runs remain available here even though the
# hourly watcher excludes them.
#
#   Usage: dryrun.sh <workflow.yaml> <run_id>
#   e.g.   dryrun.sh sanity-tests-debug.yaml 34920175153
set -euo pipefail

WATCH_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$WATCH_HOME/config.sh"          # REPO, MODEL, TT_METAL_DIR, PATH self-heal, PIPELINES

if [[ -z "$TT_METAL_DIR" || ! -d "$TT_METAL_DIR/.git" ]]; then
  echo "FATAL: TT_METAL_DIR must name a tt-metal Git checkout; set it in the host-local config or environment" >&2
  exit 1
fi
PROMPT_TEMPLATE="$WATCH_HOME/agent_prompt.txt"

WF="$1"; RID="$2"

# Resolve the run first, then select the entry for its event + display title.
# Multiple configurations intentionally share a workflow filename.
run_json=$(gh api "repos/$REPO/actions/runs/$RID")
run_event=$(jq -r '.event' <<<"$run_json")
run_title=$(jq -r '.display_title // .name // ""' <<<"$run_json")
entry=""
title_match=""
workflow_match=""
for p in "${PIPELINES[@]}"; do
  IFS='|' read -r candidate_key candidate_workflow candidate_display candidate_event candidate_pattern _rest <<<"$p"
  if [[ "$candidate_workflow" == "$WF" ]]; then
    [[ -z "$workflow_match" ]] && workflow_match="$p"
    [[ "$run_title" =~ $candidate_pattern ]] && title_match="$p"
  fi
  if [[ "$candidate_workflow" == "$WF" && "$candidate_event" == "$run_event" && "$run_title" =~ $candidate_pattern ]]; then
    entry="$p"
    break
  fi
done
if [[ -z "$entry" && "$run_event" == "workflow_dispatch" ]]; then
  # Manual runs are excluded from the hourly digest but remain inspectable.
  # Prefer the corresponding scheduled title variant; otherwise use the
  # workflow's generic kernel-lib matcher and let the actual job name provide
  # the SKU/configuration to the triage agent.
  entry="${title_match:-$workflow_match}"
fi
[[ -z "$entry" ]] && { echo "no PIPELINES configuration for $WF run '$run_title' (event=$run_event)" >&2; exit 1; }
IFS='|' read -r state_key workflow display event_filter run_pattern test_hint job_pattern <<<"$entry"
if [[ "$run_event" == "workflow_dispatch" ]]; then
  display="$display manual"
  test_hint="In-scope = the kernel lib tests job(s) in this manually dispatched run. Derive the selected SKU, platform, and debug flags from the job names; everything else is out of scope."
fi

# --- fetch job-filtered failure logs (same extraction as watch.sh) ----------
failed_jobs=$(gh api "repos/$REPO/actions/runs/$RID/jobs" --paginate \
              --jq '.jobs[] | select(.conclusion=="failure") | "\(.id)\t\(.name)"')
[[ -n "$job_pattern" ]] && failed_jobs=$(printf '%s\n' "$failed_jobs" | grep -E -i "$job_pattern" || true)
echo ">>> in-scope failed jobs:" >&2; printf '%s\n' "$failed_jobs" | cut -f2 >&2

logs=""
while IFS=$'\t' read -r jid jname; do
  [[ -z "$jid" ]] && continue
  full=$(gh api "repos/$REPO/actions/jobs/$jid/logs" 2>/dev/null)
  raw=$(printf '%s' "$full" \
    | { grep -E -B 6 -A 24 'Watcher stopped the device|tripped an assert on line|Watcher detected tripped assert|Fatal Python error|TT_FATAL|TT_ASSERT|PCC.{0,80}(fail|mismatch)|AssertionError|FAILED ' || true; })
  ci=$(printf '%s' "$full" \
    | { grep -E -B 2 -A 10 '#### Error Message|#### Root Cause|#### Suggested Action|<summary>Failed Tests' || true; })
  generic=$(printf '%s' "$full" \
    | { grep -E -B 5 -A 30 '##\[error\]|Traceback|Process completed with exit code' || true; })
  cut="${raw}${raw:+$'\n'}${ci}${ci:+$'\n'}${generic}"
  [[ -z "$cut" ]] && cut=$(printf '%s' "$full" | tail -c 12000)
  cut=$(printf '%s' "$cut" | tail -c 18000)
  logs+="=== JOB: $jname ===
$cut

"
done <<<"$failed_jobs"

# --- build the identical prompt and run the agent ---------------------------
# Pass run_number (not the run id) as "Run: #N" so the rendered header matches
# what watch.sh produces for the same run instead of showing a 11-digit id.
IFS=$'\t' read -r rnum rsha rurl rconcl < <(
  jq -r '[.run_number, .head_sha, .html_url, (.conclusion // "unknown")] | @tsv' <<<"$run_json"
)
ctx="Pipeline display name: $display
Workflow file: $workflow
Run: #$rnum  conclusion=$rconcl  sha=$rsha
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
