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

# Serialize runs: prevent overlapping invocations (a stray second cron daemon,
# or a manual run coinciding with a scheduled tick) from double-posting to Slack
# or racing on the shared ~/.claude/.credentials.json OAuth refresh. Non-blocking
# — if another instance already holds the lock, log once and exit cleanly.
exec 200>"$SDPA_HOME/.watch.lock"
if ! flock -n 200; then
  log "another watch.sh instance holds the lock — skipping this tick"
  exit 0
fi

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
  # rc=127 means the claude binary vanished mid-call — almost always its
  # self-updater swapping claude.exe while we ran. Wait out the swap and
  # retry once before giving up on this run.
  if [[ $rc -eq 127 ]]; then
    log "  claude not found (rc=127) — likely a binary swap; retrying in 20s"
    sleep 20
    summary=$(cd "$TT_METAL_DIR" && \
              claude --model "$MODEL" -p <<<"$prompt" 2>>"$AGENT_ERR")
    rc=$?
  fi
  set -e

  # opus-4.8 sometimes emits reasoning prose before the block despite the
  # prompt's "one block and nothing else" rule. Keep only from the "▸ *"
  # header line onward so neither the digest nor the success-line collapse
  # ever ingests preamble. If no header line exists the result is empty and
  # the fallback below fires (malformed output treated as an agent failure).
  if [[ $rc -eq 0 && -n "$summary" ]]; then
    summary=$(printf '%s\n' "$summary" | sed -n '/^▸/,$p')
  fi

  # Agent failure: 🟡 block (not ⚠️ — that triggers the walk-back) plus
  # the last cached summary so the digest still shows real test state.
  # 🟡 first line tells the caller to skip persisting, so next tick retries.
  if [[ $rc -ne 0 || -z "$summary" ]]; then
    log "  agent failed on #$rnum (rc=$rc); using fallback block"
    local cached
    cached=$(jq -r --arg w "$workflow" '.[$w].summary // ""' "$STATE")
    if [[ -n "$cached" ]]; then
      cached=$(printf '%s' "$cached" | sed -E '1 s/^▸ \*[^*]+\*  ?//')
    else
      cached="(none)"
    fi
    summary="▸ *$display*  🟡 agent error (rc=$rc) — run #${rnum} ($rconcl) not analyzed — $rurl
↳ last cached: $cached"
  fi
  printf '%s' "$summary"
}

# MODE B auth: refresh the interactive OAuth credential ourselves if it is at
# or near expiry. Headless `claude -p` reads $CLAUDE_CREDS_FILE but never
# refreshes it, so without this an unattended cron dies once the ~8h access
# token lapses. We POST the rotating refresh token to the OAuth endpoint and
# write the new access/refresh/expiry back atomically. Returns 0 if the
# credential is usable afterwards (fresh already, or refreshed), 1 otherwise.
refresh_oauth_credential() {
  local creds="$CLAUDE_CREDS_FILE"
  if [[ ! -f "$creds" ]]; then
    log "  no OAuth credential at $creds — log into 'claude' once to seed it, or drop a setup-token in $OAUTH_TOKEN_FILE"
    return 1
  fi
  local exp now rt resp code body at nrt ein exp_ms
  exp=$(jq -r '.claudeAiOauth.expiresAt // 0' "$creds" 2>/dev/null || echo 0)
  now=$(( $(date -u +%s) * 1000 ))
  if (( exp > now + OAUTH_REFRESH_MARGIN_SEC * 1000 )); then
    return 0   # still comfortably valid — nothing to do
  fi
  rt=$(jq -r '.claudeAiOauth.refreshToken // empty' "$creds" 2>/dev/null)
  if [[ -z "$rt" ]]; then
    log "  credential has no refresh token — cannot auto-refresh"
    return 1
  fi
  log "  OAuth access token at/near expiry — refreshing via $OAUTH_TOKEN_ENDPOINT"
  resp=$(curl -sS -w '\n__HTTP__%{http_code}' -X POST "$OAUTH_TOKEN_ENDPOINT" \
         -H 'Content-Type: application/json' \
         -d "{\"grant_type\":\"refresh_token\",\"refresh_token\":\"$rt\",\"client_id\":\"$OAUTH_CLIENT_ID\"}" \
         2>>"$AGENT_ERR")
  code=$(printf '%s' "$resp" | sed -n 's/.*__HTTP__//p')
  body=$(printf '%s' "$resp" | sed 's/__HTTP__[0-9]*$//')
  if [[ "$code" != "200" ]]; then
    log "  refresh failed (HTTP ${code:-?}) — keeping existing token; preflight will decide"
    return 1
  fi
  at=$(printf '%s' "$body" | jq -r '.access_token // empty')
  nrt=$(printf '%s' "$body" | jq -r '.refresh_token // empty')
  ein=$(printf '%s' "$body" | jq -r '.expires_in // 28800')
  if [[ -z "$at" ]]; then
    log "  refresh response missing access_token — keeping existing token"
    return 1
  fi
  exp_ms=$(( now + ein * 1000 ))
  # Merge into the credential, preserving all other fields; rotate the refresh
  # token if the server returned a new one. Atomic write + 600 perms.
  if jq --arg at "$at" --arg rt "${nrt:-$rt}" --argjson exp "$exp_ms" \
        '.claudeAiOauth.accessToken=$at | .claudeAiOauth.refreshToken=$rt | .claudeAiOauth.expiresAt=$exp' \
        "$creds" > "$creds.tmp" 2>>"$AGENT_ERR"; then
    mv "$creds.tmp" "$creds" && chmod 600 "$creds"
    log "  OAuth token refreshed (valid ~$(( ein / 3600 ))h)"
    return 0
  fi
  rm -f "$creds.tmp"
  log "  failed to write refreshed credential"
  return 1
}

# ---------- prerequisites ----------
# Auth (see config.sh "Auth" for the full rationale). Two modes, MODE A first:
#   A) a long-lived $OAUTH_TOKEN_FILE (from `claude setup-token`) → exported as
#      CLAUDE_CODE_OAUTH_TOKEN; no refresh needed.
#   B) otherwise the interactive $CLAUDE_CREDS_FILE, which we auto-refresh here
#      because headless `claude -p` won't. Zero-manual once a /login seeds it.
# A stale ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN would override either, so
# unset both defensively before every run.
unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN
OAUTH_TOKEN_FILE="${OAUTH_TOKEN_FILE:-$HOME/.sdpa-watch/oauth_token}"
if [[ -s "$OAUTH_TOKEN_FILE" ]]; then
  log "auth: MODE A (long-lived setup-token)"
  CLAUDE_CODE_OAUTH_TOKEN="$(tr -d '[:space:]' < "$OAUTH_TOKEN_FILE")"
  export CLAUDE_CODE_OAUTH_TOKEN
else
  log "auth: MODE B (auto-refreshed interactive credential)"
  refresh_oauth_credential || true   # non-fatal; preflight is the gate
fi
# Preflight so an expired/revoked/un-refreshable token fails loudly HERE (once,
# in the log) instead of silently degrading every pipeline to a 🟡 block.
if ! printf 'reply with the single word OK' \
     | claude --model "$MODEL" -p >/dev/null 2>>"$AGENT_ERR"; then
  log "FATAL: auth preflight failed — token expired/revoked and could not refresh. Re-seed via 'claude' /login (MODE B) or 'claude setup-token' → $OAUTH_TOKEN_FILE (MODE A). See $AGENT_ERR"
  exit 1
fi

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

  # 🟡 = agent error fallback; keep the old cache entry so the next tick
  # sees a cache miss and retries the analysis.
  if [[ "$(printf '%s' "$summary" | head -n1)" == *🟡* ]]; then
    log "  not persisting state for #$run_number (agent error)"
    continue
  fi

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
