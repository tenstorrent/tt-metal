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

# Extract failure markers from a failed run's job logs. Outputs a single
# multi-job blob suitable for inclusion in the agent prompt.
# Reads $job_pattern from caller's scope: if non-empty, only failed jobs
# whose .name matches (grep -E -i) the pattern have their logs fetched.
fetch_failure_logs() {
  local rid="$1"
  local failed_jobs combined jid jname jlog_full jlog
  failed_jobs=$(gh api "repos/$REPO/actions/runs/$rid/jobs" --paginate \
                --jq '.jobs[] | select(.conclusion=="failure") | "\(.id)\t\(.name)"' 2>/dev/null)
  if [[ -z "$failed_jobs" ]]; then
    printf '(no failed jobs reported)'
    return
  fi
  if [[ -n "${job_pattern:-}" ]]; then
    local before_count after_count
    before_count=$(printf '%s\n' "$failed_jobs" | grep -c .)
    failed_jobs=$(printf '%s\n' "$failed_jobs" | grep -E -i "$job_pattern" || true)
    after_count=$(printf '%s\n' "$failed_jobs" | grep -c .)
    log "  job filter '$job_pattern': $after_count/$before_count failed jobs in scope"
    if [[ -z "$failed_jobs" ]]; then
      printf '(no in-scope failed jobs — %d failed jobs filtered out by pattern /%s/)' \
             "$before_count" "$job_pattern"
      return
    fi
  fi
  combined=""
  while IFS=$'\t' read -r jid jname; do
    [[ -z "$jid" ]] && continue
    jlog_full=$(gh api "repos/$REPO/actions/jobs/$jid/logs" 2>/dev/null)
    # Long CI logs (10MB+) bury FAILED markers in the body while the last
    # 12k is post-job docker cleanup. Grep for failure patterns with
    # context; fall back to 12k tail if nothing matched.
    jlog=$(printf '%s' "$jlog_full" \
           | { grep -E -B 5 -A 100 '##\[error\]|FAILED |AssertionError|Traceback' || true; })
    [[ -z "$jlog" ]] && jlog=$(printf '%s' "$jlog_full" | tail -c 12000)
    combined+="=== JOB: $jname ===
$jlog

"
  done <<<"$failed_jobs"
  printf '%s' "${combined:-(log fetch failed)}"
}

# Run the agent on one run and emit a summary block. Reads $display,
# $workflow, $test_hint, $prev_sha from the caller's scope.
analyze_run() {
  local rid="$1" rnum="$2" rconcl="$3" rsha="$4" rurl="$5" note="$6"
  local logs="(no log fetched)" commits ctx prompt summary rc

  [[ "$rconcl" == "failure" ]] && logs=$(fetch_failure_logs "$rid")

  commits="(no prior run tracked)"
  if [[ -n "$prev_sha" && "$prev_sha" != "$rsha" ]]; then
    commits=$(git -C "$TT_METAL_DIR" log --oneline "$prev_sha..$rsha" 2>/dev/null | head -50 \
              || echo "(range unavailable)")
  fi

  ctx=$(cat <<EOF
Pipeline display name: $display
Workflow file: $workflow
Run: #$rnum  conclusion=$rconcl  sha=$rsha
URL: $rurl
Test focus hint: $test_hint
${note:+Note: $note}

Commits since last analyzed run (${prev_sha:0:7}..${rsha:0:7}):
$commits

Failure log excerpt (truncated to last ~40k chars):
$logs
EOF
)
  prompt="$(cat "$PROMPT_TEMPLATE")

# Context
$ctx"

  set +e
  summary=$(cd "$TT_METAL_DIR" && \
            claude --model "$MODEL" -p <<<"$prompt" 2>>"$AGENT_ERR")
  rc=$?
  set -e

  if [[ $rc -ne 0 || -z "$summary" ]]; then
    log "  agent failed on #$rnum (rc=$rc); using fallback block"
    summary="▸ *$display*  ❌ $rconcl  _run #${rnum}_
(agent error — see $rurl)"
  fi
  printf '%s' "$summary"
}

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
  IFS='|' read -r workflow display test_hint job_pattern <<<"$entry"
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

  # Latest run is in-flight (fresh trigger queued, or a re-attempt). The
  # cache-hit check above already handles re-attempts of the cached run.
  # Here we must look past the in-flight run to the latest *completed*
  # run: if a newer completion exists than what's cached, analyze it;
  # otherwise the cache is still the freshest real result.
  if [[ "$status" != "completed" ]]; then
    log "  run #$run_number is $status — checking latest completed"
    completed_run=$(gh api "repos/$REPO/actions/workflows/$workflow/runs?branch=$BRANCH&status=completed&per_page=1" \
                    --jq '.workflow_runs[0]' 2>/dev/null || echo "null")
    completed_id=""
    if [[ -n "$completed_run" && "$completed_run" != "null" ]]; then
      completed_id=$(jq -r '.id // empty' <<<"$completed_run")
    fi

    if [[ -n "$completed_id" && "$completed_id" != "$prev_id" ]]; then
      run="$completed_run"
      run_id="$completed_id"
      status=$(jq -r '.status // "unknown"'         <<<"$run")
      conclusion=$(jq -r '.conclusion // "unknown"' <<<"$run")
      sha=$(jq -r '.head_sha'                       <<<"$run")
      url=$(jq -r '.html_url'                       <<<"$run")
      run_number=$(jq -r '.run_number'              <<<"$run")
      log "  latest completed is #$run_number ($conclusion) — analyzing"
    else
      cached=$(jq -r --arg w "$workflow" '.[$w].summary // ""' "$STATE")
      if [[ -n "$cached" ]]; then
        blocks+=("$cached")
        log "  no newer completed run — reusing cached summary"
        continue
      fi
      blocks+=("▸ *$display* — _no completed runs on $BRANCH_")
      log "  no completed runs and no cache"
      continue
    fi
  fi

  # Cache miss: analyze the chosen run.
  log "  new run #$run_number ($conclusion) — analyzing"
  summary=$(analyze_run "$run_id" "$run_number" "$conclusion" "$sha" "$url" "")

  # If the primary run didn't actually run tests (per the agent's ⚠️
  # emoji — which covers infra setup failures as well as the GH-level
  # cancel/timeout conclusions), surface the most recent run where tests
  # *did* run so the digest still reflects real test state.
  primary_first_line=$(printf '%s' "$summary" | head -n1)
  if [[ "$primary_first_line" == *⚠️* ]]; then
    log "  primary classified ⚠️ (tests didn't run) — searching for last test-ran run"
    # Walk back through recent completed runs (newest first), skipping
    # the primary itself, runs whose GH conclusion already implies no
    # test execution, and any candidate the agent also classifies ⚠️.
    # Capped to avoid burning many agent calls when an infra outage
    # affects a streak of runs.
    fb_summary=""
    fb_checked=0
    fb_max=4
    while IFS=$'\t' read -r cid cconcl csha curl crnum; do
      [[ -z "$cid" || "$cid" == "$run_id" ]] && continue
      case "$cconcl" in
        success|failure) ;;
        *) continue ;;
      esac
      (( ++fb_checked ))
      log "  fallback candidate #$crnum ($cconcl) — analyzing"
      cand_summary=$(analyze_run "$cid" "$crnum" "$cconcl" "$csha" "$curl" \
                     "Latest completed run #$run_number did not execute tests. Analyze this earlier run as the current real test state.")
      cand_first=$(printf '%s' "$cand_summary" | head -n1)
      if [[ "$cand_first" != *⚠️* ]]; then
        fb_summary="$cand_summary"
        log "  fallback: #$crnum is the latest test-ran run"
        break
      fi
      log "  #$crnum also classified ⚠️ — continuing"
      (( fb_checked >= fb_max )) && { log "  giving up after $fb_max candidates"; break; }
    done < <(gh api "repos/$REPO/actions/workflows/$workflow/runs?branch=$BRANCH&status=completed&per_page=15" \
             --jq '.workflow_runs[] | "\(.id)\t\(.conclusion)\t\(.head_sha)\t\(.html_url)\t\(.run_number)"' 2>/dev/null)

    if [[ -n "$fb_summary" ]]; then
      # Strip the "▸ *Name*  " prefix from the fallback so the combined
      # block has a single pipeline header.
      fb_stripped=$(printf '%s' "$fb_summary" | sed -E '1 s/^▸ \*[^*]+\*  ?//')
      summary="$summary
↳ Last test-ran: $fb_stripped"
    else
      summary="$summary
↳ Last test-ran: not found within recent runs"
    fi
  fi

  blocks+=("$summary")

  # Persist new state, keyed on the primary (latest) run id.
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
