#!/usr/bin/env bash
# SDPA pipeline watcher — invoked by cron every 4h (or manually).
# Posts one Slack digest with per-pipeline status. Uses a cache keyed by
# run_id so unchanged pipelines reuse their previous summary block without
# re-invoking the LLM.
#
# Run manually with DRY_RUN=1 to print the digest instead of posting.

set -euo pipefail

SDPA_HOME="$HOME/.sdpa-watch"
source "$SDPA_HOME/config.sh"

STATE="$SDPA_HOME/state.json"
PROMPT_TEMPLATE="$SDPA_HOME/agent_prompt.txt"
AGENT_ERR="$SDPA_HOME/agent_errors.log"
DRY_RUN="${DRY_RUN:-0}"

ts()  { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(ts)] $*" >&2; }

# ---------- prerequisites ----------
[[ -f "$API_KEY_FILE" ]] || { log "FATAL: $API_KEY_FILE missing"; exit 1; }
export ANTHROPIC_API_KEY
ANTHROPIC_API_KEY="$(cat "$API_KEY_FILE")"

SLACK_URL=""
if [[ -f "$SLACK_WEBHOOK_FILE" ]]; then
  SLACK_URL="$(cat "$SLACK_WEBHOOK_FILE")"
fi
if [[ -z "$SLACK_URL" && "$DRY_RUN" != "1" ]]; then
  log "FATAL: $SLACK_WEBHOOK_FILE missing or empty (run with DRY_RUN=1 to test without Slack)"
  exit 1
fi

[[ -f "$STATE" ]] || echo '{}' > "$STATE"

# Best-effort fetch so commit-range lookups work.
git -C "$TT_METAL_DIR" fetch --quiet origin "$BRANCH" 2>/dev/null \
  || log "warn: git fetch failed; commit-range data may be stale"

# ---------- per-pipeline processing ----------
blocks=()

for entry in "${PIPELINES[@]}"; do
  IFS='|' read -r workflow display test_hint <<<"$entry"
  log "checking: $workflow ($display)"

  # Latest run on $BRANCH (any status). We deliberately do NOT pass
  # status=completed: a re-attempt flips the run back to in_progress, and
  # the filter would then hide it and surface an OLDER completed run.
  # Status is checked client-side below.
  run=$(gh api "repos/$REPO/actions/workflows/$workflow/runs?branch=$BRANCH&per_page=1" \
        --jq '.workflow_runs[0]' 2>/dev/null || echo "null")

  if [[ -z "$run" || "$run" == "null" ]]; then
    blocks+=("▸ *$display* — _no runs on $BRANCH_")
    log "  no runs found"
    continue
  fi

  run_id=$(jq -r '.id'         <<<"$run")
  status=$(jq -r '.status // "unknown"' <<<"$run")
  conclusion=$(jq -r '.conclusion // "unknown"' <<<"$run")
  sha=$(jq -r '.head_sha'      <<<"$run")
  url=$(jq -r '.html_url'      <<<"$run")
  run_number=$(jq -r '.run_number' <<<"$run")

  prev=$(jq --arg w "$workflow" '.[$w] // {}' "$STATE")
  prev_id=$(jq -r '.run_id // ""' <<<"$prev")
  prev_sha=$(jq -r '.sha    // ""' <<<"$prev")

  # Cache hit: same run as last time (covers in-progress re-attempts of the
  # cached run too — run_id is stable across attempts).
  if [[ "$run_id" == "$prev_id" ]]; then
    cached=$(jq -r --arg w "$workflow" '.[$w].summary // ""' "$STATE")
    if [[ -n "$cached" ]]; then
      blocks+=("$cached")
      log "  cache hit (run #$run_number unchanged)"
      continue
    fi
  fi

  # New run_id but not yet completed (fresh trigger queued, or a re-attempt
  # in flight). Prefer the previous cached block; if there is none, fall
  # back to the latest *completed* run so we always show real data.
  if [[ "$status" != "completed" ]]; then
    cached=$(jq -r --arg w "$workflow" '.[$w].summary // ""' "$STATE")
    if [[ -n "$cached" ]]; then
      blocks+=("$cached")
      log "  run #$run_number is $status — reusing previous cached summary"
      continue
    fi
    log "  run #$run_number is $status with no cache — falling back to latest completed"
    run=$(gh api "repos/$REPO/actions/workflows/$workflow/runs?branch=$BRANCH&status=completed&per_page=1" \
          --jq '.workflow_runs[0]' 2>/dev/null || echo "null")
    if [[ -z "$run" || "$run" == "null" ]]; then
      blocks+=("▸ *$display* — _no completed runs on $BRANCH_")
      log "  no completed runs found"
      continue
    fi
    run_id=$(jq -r '.id'              <<<"$run")
    status=$(jq -r '.status // "unknown"'     <<<"$run")
    conclusion=$(jq -r '.conclusion // "unknown"' <<<"$run")
    sha=$(jq -r '.head_sha'           <<<"$run")
    url=$(jq -r '.html_url'           <<<"$run")
    run_number=$(jq -r '.run_number'  <<<"$run")
  fi

  # Cache miss: fetch logs (only on failure) and commit range, run agent.
  log "  new run #$run_number ($conclusion) — analyzing"

  logs="(no log fetched)"
  if [[ "$conclusion" == "failure" ]]; then
    # `gh run view --log-failed` silently returns 0 bytes for some repos/runs.
    # Iterate failed jobs via the API and fetch each job's log directly.
    failed_jobs=$(gh api "repos/$REPO/actions/runs/$run_id/jobs" --paginate \
                  --jq '.jobs[] | select(.conclusion=="failure") | "\(.id)\t\(.name)"' 2>/dev/null)
    if [[ -n "$failed_jobs" ]]; then
      combined=""
      while IFS=$'\t' read -r jid jname; do
        [[ -z "$jid" ]] && continue
        jlog_full=$(gh api "repos/$REPO/actions/jobs/$jid/logs" 2>/dev/null)
        # Long CI logs (10MB+) bury FAILED markers in the body while
        # the last 12k is post-job docker cleanup. Grep for failure
        # patterns with context — output size is bounded by the number
        # of matches, not log size. Fall back to a 12k tail only if
        # nothing matched, so timeout/cleanup signal still gets through.
        jlog=$(printf '%s' "$jlog_full" \
               | { grep -E -B 5 -A 100 '##\[error\]|FAILED |AssertionError|Traceback' || true; })
        [[ -z "$jlog" ]] && jlog=$(printf '%s' "$jlog_full" | tail -c 12000)
        combined+="=== JOB: $jname ===
$jlog

"
      done <<<"$failed_jobs"
      logs="${combined:-(log fetch failed)}"
    else
      logs="(no failed jobs reported)"
    fi
  fi

  commits="(no prior run tracked)"
  if [[ -n "$prev_sha" && "$prev_sha" != "$sha" ]]; then
    commits=$(git -C "$TT_METAL_DIR" log --oneline "$prev_sha..$sha" 2>/dev/null | head -50 \
              || echo "(range unavailable)")
  fi

  context=$(cat <<EOF
Pipeline display name: $display
Workflow file: $workflow
Run: #$run_number  conclusion=$conclusion  sha=$sha
URL: $url
Test focus hint: $test_hint

Commits since last analyzed run (${prev_sha:0:7}..${sha:0:7}):
$commits

Failure log excerpt (truncated to last ~40k chars):
$logs
EOF
)

  full_prompt="$(cat "$PROMPT_TEMPLATE")

# Context
$context"

  # Pass the prompt over stdin, not via `-p "$full_prompt"`. Inlining a
  # large prompt as an argv arg blows past ARG_MAX (~128 KB) when failure
  # logs are big and the kernel rejects the exec with E2BIG.
  set +e
  summary=$(cd "$TT_METAL_DIR" && \
            claude --model "$MODEL" -p <<<"$full_prompt" 2>>"$AGENT_ERR")
  rc=$?
  set -e

  if [[ $rc -ne 0 || -z "$summary" ]]; then
    log "  agent failed (rc=$rc); using fallback block"
    summary="▸ *$display*  ❌ $conclusion  _run #${run_number}_
(agent error — see $url)"
  fi

  blocks+=("$summary")

  # Persist new state.
  jq --arg w "$workflow" --arg id "$run_id" --arg sha "$sha" --arg sm "$summary" \
     '.[$w] = {run_id: $id, sha: $sha, summary: $sm, updated: now}' \
     "$STATE" > "$STATE.tmp" && mv "$STATE.tmp" "$STATE"
done

# ---------- assemble digest (Slack Block Kit) ----------
# Belgrade = Central European Time with EU DST rules. POSIX TZ string used
# instead of "Europe/Belgrade" because this host has no tzdata installed.
ts_human="$(TZ='CET-1CEST,M3.5.0,M10.5.0/3' date +'%Y-%m-%d %H:%M %Z')"
title="SDPA Pipelines — $BRANCH — $ts_human"

# Split into success (collapse to one line) and failure (keep full block).
success_names=()
failure_blocks=()
for b in "${blocks[@]}"; do
  first_line=$(printf '%s' "$b" | head -n1)
  if [[ "$first_line" == *✅* ]]; then
    name=$(printf '%s' "$first_line" | sed -E 's/^▸ \*([^*]+)\*.*/\1/')
    success_names+=("$name")
  else
    failure_blocks+=("$b")
  fi
done

success_line=""
if [[ ${#success_names[@]} -gt 0 ]]; then
  joined=$(printf ', %s' "${success_names[@]}")
  joined="${joined:2}"
  success_line="✅ $joined"
fi

payload=$(jq -nc \
  --arg title "$title" \
  --arg succ "$success_line" \
  --args \
  '{
     text: $title,
     blocks: (
       [{type: "header", text: {type: "plain_text", text: $title, emoji: true}}]
       + (if $succ != "" then [{type: "section", text: {type: "mrkdwn", text: $succ}}] else [] end)
       + ($ARGS.positional | map([{type: "divider"},
                                  {type: "section", text: {type: "mrkdwn", text: .}}]) | add // [])
     )
   }' \
  "${failure_blocks[@]+"${failure_blocks[@]}"}")

# ---------- post or dry-run ----------
if [[ "$DRY_RUN" == "1" ]]; then
  log "DRY RUN — would post to Slack:"
  echo "================================================================"
  echo "$title"
  [[ -n "$success_line" ]] && echo "$success_line"
  for b in "${failure_blocks[@]+"${failure_blocks[@]}"}"; do
    echo "----------------------------------------------------------------"
    printf '%s\n' "$b"
  done
  echo "================================================================"
else
  resp=$(printf '%s' "$payload" \
         | curl -sS -X POST -H 'Content-Type: application/json' --data @- "$SLACK_URL")
  if [[ "$resp" == "ok" ]]; then
    log "posted to Slack (${#payload} chars JSON)"
  else
    log "WARN: Slack response: $resp"
  fi
fi
