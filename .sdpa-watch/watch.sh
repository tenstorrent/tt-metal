#!/usr/bin/env bash
# SDPA + Kimi K3 pipeline watcher — invoked by cron every 4h (or manually).
# Posts one Slack digest with per-pipeline status. Uses a cache keyed by
# run_id so unchanged pipelines reuse their previous summary block without
# re-invoking the LLM.
#
# Run manually with DRY_RUN=1 to print the digest instead of posting.

set -euo pipefail

SDPA_HOME="$HOME/.sdpa-watch"
source "$SDPA_HOME/config.sh"

STATE="${WATCH_STATE_FILE:-$SDPA_HOME/state.json}"   # override = dry-run on a copy
PROMPT_TEMPLATE="$SDPA_HOME/agent_prompt.txt"
AGENT_ERR="$SDPA_HOME/agent_errors.log"
DRY_RUN="${DRY_RUN:-0}"

# ---------- logging ----------
# One file per UTC day under logs/, so "show me 1 September" is a single file
# instead of a grep through 50k lines.
#
# Why each instance buffers locally first: $HOME is NFS4, where O_APPEND is
# NOT atomic, and this box runs two cron daemons (`cron -f` and `cron -P`) that
# both fire the crontab entry — so ~4 instances raced per tick, all appending
# to one shared file. That shredded it: interleaved half-lines ("ick" on its
# own) and 29k NUL bytes by 2026-09-01. Buffering to a LOCAL temp file and
# flushing once, under flock, makes each tick one contiguous ordered chunk.
LOG_DIR="$SDPA_HOME/logs"
LOG_RETENTION_DAYS="${LOG_RETENTION_DAYS:-60}"
mkdir -p "$LOG_DIR"

# Interactive runs and DRY_RUN keep their output on the terminal, so a manual
# preview never lands in the day log.
if [[ "$DRY_RUN" != "1" && ! -t 1 ]]; then
  RUN_LOG="$(mktemp "${TMPDIR:-/tmp}/watch-$$-XXXXXX.log")"
  exec >>"$RUN_LOG" 2>&1
  flush_run_log() {
    local day
    day="$(date -u +%F)"
    flock "$LOG_DIR/.write.lock" -c "cat '$RUN_LOG' >> '$LOG_DIR/$day.log'" \
      2>/dev/null || cat "$RUN_LOG" >> "$LOG_DIR/$day.log"
    # today.log always points at the current day, so `tail -F today.log`
    # survives midnight without knowing the date.
    ln -sfn "$day.log" "$LOG_DIR/today.log" 2>/dev/null || true
    rm -f "$RUN_LOG"
  }
  trap flush_run_log EXIT
fi

ts()  { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(ts)] $*" >&2; }

# Local wall-clock for anything a human reads in Slack. Belgrade = Central
# European Time with EU DST rules; POSIX TZ string instead of "Europe/Belgrade"
# because this host has no tzdata installed.
ts_local() { TZ='CET-1CEST,M3.5.0,M10.5.0/3' date +'%Y-%m-%d %H:%M %Z'; }

# Serialize runs: prevent overlapping invocations (a stray second cron daemon,
# or a manual run coinciding with a scheduled tick) from double-posting to Slack
# or racing on the shared ~/.claude/.credentials.json OAuth refresh. Non-blocking
# — if another instance already holds the lock, log once and exit cleanly.
exec 200>"$SDPA_HOME/.watch.lock"
if ! flock -n 200; then
  log "another watch.sh instance holds the lock — skipping this tick"
  exit 0
fi

# Winner only, once per tick: drop day logs past the retention window.
find "$SDPA_HOME/joblogs" -maxdepth 1 -name '*.txt' -type f -mtime +7 -delete 2>/dev/null || true
find "$LOG_DIR" -maxdepth 1 -name '20*-*-*.log' -type f \
     -mtime "+$LOG_RETENTION_DAYS" -delete 2>/dev/null || true

# Failure excerpt of ONE finished job, cached on disk by job id. A finished
# job's log never changes (a re-run attempt is a new job id), so each job is
# downloaded once however many ticks — or the autofix bot — ask for it.
# Long CI logs (10MB+) bury FAILED markers in the body while the last 12k is
# post-job docker cleanup: grep for failure patterns with context, fall back
# to the 12k tail if nothing matched.
job_log_excerpt() {
  local jid="$1" dir="${SDPA_HOME:-$HOME/.sdpa-watch}/joblogs" full ex
  mkdir -p "$dir"
  if [[ -s "$dir/$jid.txt" ]]; then cat "$dir/$jid.txt"; return; fi
  full=$(gh api "repos/$REPO/actions/jobs/$jid/logs" 2>/dev/null | tr -d '\000') || full=""
  ex=$(printf '%s' "$full" | { grep -E -B 5 -A 100 '##\[error\]|FAILED |AssertionError|Traceback' || true; })
  [[ -z "$ex" ]] && ex=$(printf '%s' "$full" | tail -c 12000)
  [[ -n "$full" ]] && printf '%s' "$ex" > "$dir/$jid.txt"
  printf '%s' "$ex"
}

# In-scope jobs of a run, ANY status, as [{id, name, status, conclusion}].
# Reads $job_pattern from the caller (empty = every job). The API returns the
# latest attempt of each job, so a re-run replaces the attempt it re-ran.
# Three tries: L2's run has several pages of jobs, so a transient 502 or TLS
# timeout on one page is common enough to cost a pipeline its tick.
inscope_jobs() {
  local i out
  for i in 1 2 3; do
    if out=$(gh api "repos/$REPO/actions/runs/$1/jobs?per_page=100" --paginate \
               --jq '.jobs[] | {id, name, status, conclusion}' 2>>"$AGENT_ERR" \
             | jq -sc --arg p "${job_pattern:-}" '[.[] | select($p == "" or (.name | test($p; "i")))]'); then
      printf '%s' "$out"; return 0
    fi
    sleep $((i * 5))
  done
  return 1
}

# While a run is in progress, append its job progress to the block header.
# Reads $partial / $done_n / $total_n from the caller.
with_progress() {
  if (( partial )); then
    printf '%s' "$1" | sed "1 s|\$|  ⏳ $done_n/$total_n in-scope jobs done|"
  else
    printf '%s' "$1"
  fi
}

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
    jlog=$(job_log_excerpt "$jid")
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
  # Rules learned from past mistakes, shared with the autofix bot
  # (~/.sdpa-fix/LESSONS.md, lines tagged "watcher").
  local lessons=""
  if [[ -f "$HOME/.sdpa-fix/LESSONS.md" ]]; then
    lessons=$(grep -E '^- \[[a-z,]*\bwatcher\b[a-z,]*\] ' "$HOME/.sdpa-fix/LESSONS.md" | sed -E 's/^- \[[a-z,]+\] /- /')
    [[ -n "$lessons" ]] && lessons=$'\n\n# Lessons from past mistakes (follow these)\n'"$lessons"
  fi
  prompt="$(cat "$PROMPT_TEMPLATE")$lessons

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
SLACK_BOT_TOKEN=""
if [[ -n "${SLACK_BOT_TOKEN_FILE:-}" && -f "$SLACK_BOT_TOKEN_FILE" ]]; then
  SLACK_BOT_TOKEN="$(tr -d '[:space:]' < "$SLACK_BOT_TOKEN_FILE")"
fi
if [[ -n "$SLACK_BOT_TOKEN" && -n "${SLACK_CHANNEL_ID:-}" ]]; then
  log "slack: bot-API mode — edit-in-place digest in $SLACK_CHANNEL_ID"
elif [[ -n "$SLACK_URL" ]]; then
  log "slack: webhook mode — new message per tick (no bot token/channel configured)"
elif [[ "$DRY_RUN" != "1" ]]; then
  log "FATAL: no Slack credentials — need $SLACK_BOT_TOKEN_FILE + SLACK_CHANNEL_ID, or $SLACK_WEBHOOK_FILE (run with DRY_RUN=1 to test without Slack)"
  exit 1
fi

[[ -f "$STATE" ]] || echo '{}' > "$STATE"

# Debounce duplicate invocations: two cron daemons run on this host, so each
# scheduled tick fires 2-3×. The flock above only SERIALIZES them — a
# duplicate that starts after the first finishes still reruns the whole tick,
# and its API burst right on the heels of the healthy run's is exactly what
# trips GitHub's secondary rate limit (the all-"no runs" digest flap).
# FORCE=1 overrides for deliberate back-to-back manual runs.
if [[ "${FORCE:-0}" != "1" && "$DRY_RUN" != "1" ]]; then
  last_end=$(jq -r '._last_tick_end // 0' "$STATE" 2>/dev/null || echo 0)
  now_epoch=$(date +%s)
  if (( now_epoch - last_end < 600 )); then
    log "duplicate-tick debounce — previous tick finished $((now_epoch - last_end))s ago (<600s); FORCE=1 to override"
    exit 0
  fi
fi

# Best-effort fetch so commit-range lookups work.
git -C "$TT_METAL_DIR" fetch --quiet origin "$BRANCH" 2>/dev/null \
  || log "warn: git fetch failed; commit-range data may be stale"

# GitHub's run list serves a stale index for branch=/event= filtered queries
# (2026-10: L2's "latest" main run came back as September's #9871 for days,
# freezing the digest behind the stale-page guard). Adding a created>= window
# routes the query to a fresh index. 10 days covers every watched schedule.
RECENT="&created=>$(date -u -d '-10 days' +%F)"

# ---------- per-pipeline processing ----------
blocks=()
api_failures=0

for entry in "${PIPELINES[@]}"; do
  IFS='|' read -r workflow display test_hint job_pattern <<<"$entry"
  log "checking: $workflow ($display)"

  # Optional per-workflow event filter (see PIPELINE_EVENT in config.sh):
  # nightlies whose manual workflow_dispatch re-runs interleave with the
  # schedule get pinned to event=schedule so a green manual run can never
  # mask a still-failing scheduled nightly.
  event_filter="${PIPELINE_EVENT[$workflow]:-}"
  evq="${event_filter:+&event=$event_filter}"
  [[ -n "$event_filter" ]] && log "  event filter: $event_filter"

  # Latest run on $BRANCH (any status). Status is checked client-side below:
  # a re-attempt flips a run back to in_progress, and with per-job tracking an
  # in-progress run is reported from its FINISHED in-scope jobs.
  # per_page=10 + newest created_at instead of trusting item [0] of a
  # per_page=1 response: the list endpoint occasionally serves a stale page
  # whose first item is an ancient run (2026-08-19 it returned May's #6154
  # as L2's "latest", which then got cached as current state).
  if ! run=$(gh api "repos/$REPO/actions/workflows/$workflow/runs?branch=$BRANCH$RECENT&per_page=10$evq" \
        --jq '[.workflow_runs[]] | sort_by(.created_at) | last' 2>>"$AGENT_ERR"); then
    # gh failed (burst rate limit / auth / network) — NOT the same as "no
    # runs". Reuse the cached summary so the digest content (and therefore
    # its fingerprint) stays stable instead of flapping to a bogus
    # all-"no runs" digest; only surface an error block when there is no
    # cache to fall back on. Root-cause error text lands in agent_errors.log.
    api_failures=$((api_failures + 1))
    cached=$(jq -r --arg w "$workflow" '.[$w].summary // ""' "$STATE")
    if [[ -n "$cached" ]]; then
      blocks+=("$cached")
      log "  WARN: gh api failed — reusing cached summary (see agent_errors.log)"
    else
      blocks+=("▸ *$display* — ⚠️ _GitHub query failed this tick_")
      log "  WARN: gh api failed — no cache to fall back on"
    fi
    continue
  fi

  if [[ -z "$run" || "$run" == "null" ]]; then
    blocks+=("▸ *$display* — _no runs on ${BRANCH}_")
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
  prev_num=$(jq -r '.run_number // 0' <<<"$prev")
  prev_key=$(jq -r '.cache_key // ""' <<<"$prev")
  cached=$(jq -r '.summary // ""' <<<"$prev")
  # Progress of the cached block, for the paths that only re-show it.
  partial=$(jq -r 'if .partial == true then 1 else 0 end' <<<"$prev")
  done_n=$(jq -r '.done // 0' <<<"$prev"); total_n=$(jq -r '.total // 0' <<<"$prev")

  # Stale-page guard: run_number is monotonic per workflow, so a chosen run
  # OLDER than the cached one can only mean the API served a stale page (it
  # cannot be a genuinely new run). Reuse the cache; next tick retries.
  if (( prev_num > 0 && run_number < prev_num )) && [[ -n "$cached" ]]; then
    blocks+=("$(with_progress "$cached")")
    log "  stale API page (run #$run_number < cached #$prev_num) — reusing cached summary"
    continue
  fi

  if ! jobs=$(inscope_jobs "$run_id"); then
    api_failures=$((api_failures + 1))
    if [[ -n "$cached" ]]; then
      blocks+=("$(with_progress "$cached")"); log "  WARN: job list failed — reusing cached summary"
    else
      blocks+=("▸ *$display* — ⚠️ _GitHub query failed this tick_")
    fi
    continue
  fi
  done_n=$(jq '[.[] | select(.status == "completed")] | length' <<<"$jobs")

  # Nothing in scope of the newest run has finished yet: report the latest
  # completed run instead (or keep the cache if that is what it already is).
  if [[ "$status" != "completed" && "$done_n" == "0" ]]; then
    log "  run #$run_number is $status with 0/$(jq length <<<"$jobs") in-scope jobs done — checking latest completed"
    completed_run=$(gh api "repos/$REPO/actions/workflows/$workflow/runs?branch=$BRANCH$RECENT&status=completed&per_page=10$evq" \
                    --jq '[.workflow_runs[]] | sort_by(.created_at) | last' 2>/dev/null || echo "null")
    completed_id=""
    if [[ -n "$completed_run" && "$completed_run" != "null" ]]; then
      completed_id=$(jq -r '.id // empty' <<<"$completed_run")
    fi
    if [[ -n "$completed_id" && "$completed_id" != "$prev_id" ]] \
       && (( $(jq -r '.run_number' <<<"$completed_run") >= prev_num )); then
      run="$completed_run"
      run_id="$completed_id"
      status=$(jq -r '.status // "unknown"'         <<<"$run")
      conclusion=$(jq -r '.conclusion // "unknown"' <<<"$run")
      sha=$(jq -r '.head_sha'                       <<<"$run")
      url=$(jq -r '.html_url'                       <<<"$run")
      run_number=$(jq -r '.run_number'              <<<"$run")
      if ! jobs=$(inscope_jobs "$run_id"); then
        api_failures=$((api_failures + 1))
        [[ -n "$cached" ]] && blocks+=("$(with_progress "$cached")") || blocks+=("▸ *$display* — ⚠️ _GitHub query failed this tick_")
        continue
      fi
      log "  latest completed is #$run_number ($conclusion)"
    else
      partial=$(jq -r 'if .partial == true then 1 else 0 end' <<<"$prev")
      done_n=$(jq -r '.done // 0' <<<"$prev")
      if [[ -n "$cached" ]]; then
        blocks+=("$(with_progress "$cached")")
        log "  no newer completed run — reusing cached summary"
        continue
      fi
      blocks+=("▸ *$display* — _no completed runs on ${BRANCH}_")
      log "  no completed runs and no cache"
      continue
    fi
  fi

  partial=0; [[ "$status" != "completed" ]] && partial=1
  total_n=$(jq 'length' <<<"$jobs")
  done_n=$(jq '[.[] | select(.status == "completed")] | length' <<<"$jobs")
  finished=$(jq -c '[.[] | select(.status == "completed") | {id, name, conclusion}]' <<<"$jobs")
  fail_ids=$(jq -r '[.[] | select(.status == "completed" and .conclusion == "failure") | .id | tostring] | sort | join(",")' <<<"$jobs")
  failed_names=$(jq -c '[.[] | select(.status == "completed" and .conclusion == "failure") | .name] | unique' <<<"$jobs")

  # Baseline = the last COMPLETED run reported for this pipeline. While a new
  # run is in progress, jobs that failed in the baseline and have not finished
  # yet are "carried": the block must not flip to ✅ just because the job that
  # failed last night is still queued.
  if [[ "$run_id" == "$prev_id" || "$(jq -r '.partial == true' <<<"$prev")" == "true" ]]; then
    base_failed=$(jq -c '.base_failed // []' <<<"$prev")
    base_summary=$(jq -r '.base_summary // ""' <<<"$prev")
  else
    base_failed=$(jq -c '.failed_names // []' <<<"$prev")
    base_summary="$cached"
  fi
  carry="[]"
  if (( partial )); then
    carry=$(jq -c --argjson b "$base_failed" --argjson f "$finished" '$b - [$f[].name]' <<<"null")
  fi
  cache_key="$run_id|$( ((partial)) && echo p || echo c )|$fail_ids|$(jq -r 'join(",")' <<<"$carry")"

  # Same run, same failed jobs, same carried jobs → the cached analysis still
  # holds. Only the progress counter and the finished-jobs list move on.
  if [[ "$run_id" == "$prev_id" && "$cache_key" == "$prev_key" && -n "$cached" ]]; then
    blocks+=("$(with_progress "$cached")")
    jq --arg w "$workflow" --argjson f "$finished" --argjson d "$done_n" --argjson t "$total_n" \
       '.[$w].jobs = $f | .[$w].done = $d | .[$w].total = $t' "$STATE" > "$STATE.tmp" && mv "$STATE.tmp" "$STATE"
    log "  cache hit (run #$run_number, $done_n/$total_n in-scope jobs done)"
    continue
  fi

  carry_line=""
  if [[ "$carry" != "[]" ]]; then
    carry_line="↳ still pending in run #$run_number (failed in the last completed run): $(jq -r 'map("`" + . + "`") | join(", ")' <<<"$carry")"
  fi

  if (( partial )) && [[ -z "$fail_ids" ]]; then
    if [[ -n "$carry_line" && -n "$base_summary" ]]; then
      summary="$base_summary
$carry_line"
      log "  run #$run_number in progress, no new failures; $(jq length <<<"$carry") previously-failing job(s) not finished — keeping last result"
    else
      summary="▸ *$display*  ✅ success  _run #${run_number}_
$url"
      log "  run #$run_number in progress, all $done_n finished in-scope jobs green — no agent call"
    fi
  else
    if (( partial )); then
      log "  run #$run_number in progress ($done_n/$total_n in-scope jobs done, failed: $fail_ids) — analyzing"
      summary=$(analyze_run "$run_id" "$run_number" "failure" "$sha" "$url" \
                "This run is STILL IN PROGRESS: $done_n of $total_n in-scope jobs have finished. Judge only the finished jobs; jobs that have not finished are NOT failures and must not be reported.")
      [[ -n "$carry_line" ]] && summary="$summary
$carry_line"
    elif [[ -n "$fail_ids" && "$conclusion" != "failure" ]]; then
      # e.g. Blaze: a slow leg gets cancelled, the run concludes "cancelled",
      # yet in-scope jobs ran and failed. Judge those jobs, not the run label.
      log "  new run #$run_number ($conclusion, but in-scope jobs failed: $fail_ids) — analyzing"
      summary=$(analyze_run "$run_id" "$run_number" "failure" "$sha" "$url" \
                "The run's overall conclusion is '$conclusion' (another leg was cancelled or timed out), but $(jq length <<<"$failed_names") in-scope job(s) ran to completion and FAILED. Judge those failed jobs on their logs; do not report tests-did-not-run for them.")
    else
      log "  new run #$run_number ($conclusion) — analyzing"
      summary=$(analyze_run "$run_id" "$run_number" "$conclusion" "$sha" "$url" "")
    fi
  fi

  # If the primary run didn't actually run tests (per the agent's ⚠️
  # emoji — which covers infra setup failures as well as the GH-level
  # cancel/timeout conclusions), surface the most recent run where tests
  # *did* run so the digest still reflects real test state.
  primary_first_line=$(printf '%s' "$summary" | head -n1)
  if (( ! partial )) && [[ "$primary_first_line" == *⚠️* ]]; then
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
    done < <(gh api "repos/$REPO/actions/workflows/$workflow/runs?branch=$BRANCH$RECENT&status=completed&per_page=15$evq" \
             --jq '[.workflow_runs[]] | sort_by(.created_at) | reverse | .[] | "\(.id)\t\(.conclusion)\t\(.head_sha)\t\(.html_url)\t\(.run_number)"' 2>/dev/null)

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

  blocks+=("$(with_progress "$summary")")

  # 🟡 = agent error fallback; keep the old cache entry so the next tick
  # sees a cache miss and retries the analysis.
  if [[ "$(printf '%s' "$summary" | head -n1)" == *🟡* ]]; then
    log "  not persisting state for #$run_number (agent error)"
    continue
  fi

  # A completed run becomes the new baseline for the next in-progress run.
  if (( ! partial )); then
    base_failed="$failed_names"; base_summary="$summary"
  else
    failed_names=$(jq -c --argjson c "$carry" '. + $c | unique' <<<"$failed_names")
  fi

  # Persist new state, keyed on the reported run. run_number feeds the
  # stale-page guard; jobs (finished in-scope jobs) feeds the autofix bot.
  jq --arg w "$workflow" --arg id "$run_id" --arg sha "$sha" --arg sm "$summary" \
     --argjson num "$run_number" --arg key "$cache_key" --argjson part "$( ((partial)) && echo true || echo false )" \
     --argjson d "$done_n" --argjson t "$total_n" --argjson f "$finished" --arg u "$url" \
     --argjson fn "$failed_names" --argjson bf "$base_failed" --arg bs "$base_summary" \
     '.[$w] = {run_id: $id, run_number: $num, sha: $sha, url: $u, summary: $sm, cache_key: $key,
               partial: $part, done: $d, total: $t, jobs: $f, failed_names: $fn,
               base_failed: $bf, base_summary: $bs, updated: now}' \
     "$STATE" > "$STATE.tmp" && mv "$STATE.tmp" "$STATE"
done

# Total API outage → nothing was actually checked this tick, but every entry in
# `blocks` is already a cached echo of the last good digest (the "reusing
# cached summary" branch above), so there IS still something true to show.
# Re-render it in place, flagged stale and stamped with the LAST SUCCESSFUL
# check's time, rather than going silent: silence is indistinguishable from
# "all green, nothing changed", because chat.update never bumps a message in
# the channel. A dead gh token therefore read as a healthy digest for 30 h on
# 2026-08-30 before anyone noticed.
stale=0
fail_streak=0
if (( api_failures >= ${#PIPELINES[@]} )); then
  stale=1
  fail_streak=$(( $(jq -r '._slack.fail_streak // 0' "$STATE") + 1 ))
  log "WARN: all ${#PIPELINES[@]} GitHub queries failed this tick — re-rendering cached digest as stale (streak $fail_streak)"
  # Every block is the no-cache placeholder → genuinely nothing to say. Stay
  # silent, as before, rather than posting a wall of "query failed".
  have_cache=0
  for b in "${blocks[@]+"${blocks[@]}"}"; do
    [[ "$b" == *"_GitHub query failed this tick_"* ]] || { have_cache=1; break; }
  done
  if (( ! have_cache )); then
    log "  no cached digest to re-render — skipping Slack post"
    exit 0
  fi
fi

# ---------- assemble digest (Slack Block Kit) ----------
ts_human="$(ts_local)"

# last_ok = when the pipeline data was actually fetched. On a stale tick the
# title must carry THAT, not now, so the digest never claims a check it did not
# make. Older state predates last_ok, so fall back to the last "checked:" entry.
last_ok=$(jq -r '._slack.last_ok // (._slack.ticks[-1] // "")' "$STATE")
if (( stale )); then
  title_ts="${last_ok:-unknown}"
else
  title_ts="$ts_human"
  last_ok="$ts_human"
fi
title="SDPA + Kimi K3 Pipelines — $BRANCH — $title_ts"

# Autofix status, written INTO the failing pipeline's block: each note is
# appended to the bullet of the failing test it is about (matched on the test
# function name), e.g.
#   • `runtime:test_x [wh_n150]` — device hang … — 📌 already fixed on main by #59127 (e5f9d3186e), not in this run yet
# A tracked test that is not failing in the shown run gets no note at all;
# the ✅ verified reply in the thread covers that case. Reads the sibling
# fixer's ledger (~/.sdpa-fix); a missing ledger changes nothing.
AUTOFIX_LEDGER="$HOME/.sdpa-fix/ledger.json"
autofix_annotate() {
  local display="$1" block="$2" wf="" e out
  [[ -f "$AUTOFIX_LEDGER" ]] || { printf '%s' "$block"; return 0; }
  for e in "${PIPELINES[@]}"; do
    [[ "$(cut -d'|' -f2 <<<"$e")" == "$display" ]] && { wf="${e%%|*}"; break; }
  done
  [[ -n "$wf" ]] || { printf '%s' "$block"; return 0; }
  out=$(jq -r --arg w "$wf" --arg b "$block" --arg gh "https://github.com/$REPO" \
              --arg eo "${EMOJI_PR_OPENED:-🛠️}" --arg em "${EMOJI_PR_MERGED:-🟣}" '
    def prlink: if . == null or . == "" then "" else "<\(.)|#\(split("/") | last)>" end;
    def commitlink($sha): if ($sha // "") == "" then "" else "<\($gh)/commit/\($sha)|\($sha[0:10])>" end;
    def fixref: ((.reason // "") | (capture("#(?<n>[0-9]{4,6})") // {}) | .n) as $n
                | ([ (if $n then "<\($gh)/pull/\($n)|#\($n)>" else empty end),
                     (commitlink(.fix_sha) | select(. != "")) ] | join(" "));
    # Distinctive words, for matching a ledger record to a digest bullet when
    # their names differ (the watcher and the triage agent label separately).
    def toks: ascii_downcase | [scan("[a-z0-9_]{4,}")] | unique
              - ["test","tests","infra","kimi","with","from","that","this","failed","failure","error",
                 "sdpa","device","model","owner","runner","none","than","band","into","only","never"];
    def blabel: (((capture("^• `(?<l>[^`]*)`") // {}) | .l) // "");
    [ .sigs[] | select(.workflow == $w)
      | {f: (.test | split("::") | last | split("[") | first),
         short: (.test | split("::") | last), r: ((.last_seen.number // "") | tostring),
         tk: ("\(.test) \(.job) \(.summary // "") \(.error_key // "")" | toks),
         n: (if .state == "pr_open" then "\($eo) *autofix draft \(.pr.url | prlink) — needs your review* (targeted CI running)"
             elif .state == "ci_passed" then "\($eo) *autofix draft \(.pr.url | prlink) — CI ✅, needs your review*"
             elif .state == "ci_failed" then "\($eo) *autofix draft \(.pr.url | prlink) — CI ❌, needs your look*"
             elif .state == "ready_to_merge" then "\($eo) *autofix \(.pr.url | prlink) — ready to merge*"
             elif .state == "merged" then "\($em) autofix \(.pr.url | prlink) merged, not in this run yet"
             elif .state == "fixed_upstream" then "\($em) already fixed on main by \(fixref)\(if (.fix_author // "") != "" then " by @\(.fix_author)" else "" end), not in this run yet"
             elif .state == "fix_pending" then "\($eo) fix in progress\(if (.fix_author // "") != "" then " by @\(.fix_author)" else "" end): open PR \(.fix_pr.url | prlink)"
             elif .state == "proposed_dryrun" then "🛠 *autofix dry-run proposal — take a look*: \((.verdict_title // "see proposals/") | sub("^\\[autofix[^]]*\\] *"; ""))"
             elif .state == "awaiting_decision" then "❓ *autofix — needs your decision* (buttons in the thread)"
             elif .state == "decided" then "🛠 decision taken, autofix draft on its way"
             elif .state == "no_fix" then "🛠 no safe autofix"
             else null end)}
      | select(.n != null and (.f | length) > 3) ] as $notes
    | (($b | capture("_run #(?<n>[0-9]+)_") // {}) | .n // "") as $rn
    | ($b | split("\n")) as $lines
    | [ $lines | to_entries[] | select(.value | startswith("• "))
        | {i: .key, l: (.value | blabel), tk: (.value | toks)} ] as $bul
    # Each note goes to: the bullet(s) whose label contains the test name;
    # else, for a failure of THIS run, the bullet sharing the most distinctive
    # words (>= 3); else its own ↳ line. Failures not in the shown run get none.
    | [ $notes[] | . as $nt
        | ([ $bul[] | select((.l | length) > 0 and (.l | contains($nt.f))) | .i ]) as $exact
        | if ($exact | length) > 0 then {n: $nt.n, at: $exact}
          elif $nt.r == $rn and $rn != "" then
            ([ $bul[] | {i, ov: ([.tk[] as $t | $nt.tk | index($t) | select(. != null)] | length)} ]
             | max_by(.ov) // {ov: 0}) as $best
            | if $best.ov >= 3 then {n: $nt.n, at: [$best.i]} else {n: $nt.n, at: [], extra: "↳ `\($nt.short)` — \($nt.n)"} end
          else empty end ] as $plan
    | ($lines | to_entries
       | map(.key as $k | .value as $v
             | ([ $plan[] | select(.at | index($k)) | .n ] | unique) as $m
             | if ($m | length) > 0 then $v + " — " + ($m | join(" · ")) else $v end)) as $out
    | ([ $plan[] | .extra // empty ] | unique) as $extra
    | (if ($extra | length) > 0 and ($out | length) > 0 and ($out[-1] | startswith("http"))
       then $out[:-1] + $extra + [$out[-1]] else $out + $extra end)
    | join("\n")' "$AUTOFIX_LEDGER" 2>>"${AGENT_ERR:-/dev/null}") && [[ -n "$out" ]] || out="$block"
  printf '%s' "$out"
}

# Split into success (collapse to one line) and failure (keep full block).
success_names=()
failure_blocks=()
for b in "${blocks[@]}"; do
  first_line=$(printf '%s' "$b" | head -n1)
  if [[ "$first_line" == *✅* ]]; then
    name=$(printf '%s' "$first_line" | sed -E 's/^▸ \*([^*]+)\*.*/\1/')
    prog=$(printf '%s' "$first_line" | grep -oE '⏳ [0-9]+/[0-9]+' || true)
    success_names+=("$name${prog:+ ($prog)}")
  else
    name=$(printf '%s' "$first_line" | sed -E 's/^▸ \*([^*]+)\*.*/\1/')
    failure_blocks+=("$(autofix_annotate "$name" "$b")")
  fi
done

success_line=""
if [[ ${#success_names[@]} -gt 0 ]]; then
  joined=$(printf ', %s' "${success_names[@]}")
  joined="${joined:2}"
  success_line="✅ $joined"
fi

# ---------- status fingerprint & tick history (bot-API mode) ----------
# Fingerprint = digest content minus timestamps. Same fingerprint as the last
# posted message → same status → chat.update that message in place, appending
# this tick's time to the "checked:" history line. Different fingerprint →
# post a brand-new message (so status CHANGES still notify) with a fresh tick
# list. History lives in state.json under _slack.
# Job progress ("⏳ 6/9 …") moves every tick while a run is in flight; it is
# stripped here so a progress tick edits the digest in place instead of
# posting a new message. A new failure or a finished run still changes it.
fingerprint=$(printf '%s\n' "$success_line" "${failure_blocks[@]+"${failure_blocks[@]}"}" \
              | sed -E 's/ *\(?⏳ [0-9]+\/[0-9]+( in-scope jobs done)?\)?//g' \
              | sha256sum | awk '{print $1}')

slack_mode="webhook"
[[ -n "$SLACK_BOT_TOKEN" && -n "${SLACK_CHANNEL_ID:-}" ]] && slack_mode="bot"

msg_ts=""
ticks_json="[]"
if [[ "$slack_mode" == "bot" ]]; then
  same=$(jq -r --arg ch "$SLACK_CHANNEL_ID" --arg fp "$fingerprint" \
         '(._slack.channel // "") == $ch and (._slack.fingerprint // "") == $fp' "$STATE")
  if [[ "$same" == "true" ]]; then
    msg_ts=$(jq -r '._slack.ts // ""' "$STATE")
    ticks_json=$(jq -c '._slack.ticks // []' "$STATE")
  fi
  # Append this tick; keep the last 48 so a week-long steady state can't
  # blow past Slack's 3000-char section limit. A stale tick checked nothing, so
  # it must not enter the "checked:" history — the banner reports it instead.
  (( stale )) || ticks_json=$(jq -c --arg t "$ts_human" '(. + [$t]) | .[-48:]' <<<"$ticks_json")
fi
ticks_line=$(jq -r 'if length > 1 then "checked: " + join("  ·  ") else "" end' <<<"$ticks_json")

# The one line that makes a frozen digest legible at a glance. Deliberately a
# section (not a context) block: full-size text, first thing under the header.
stale_banner=""
if (( stale )); then
  noun="ticks"; (( fail_streak == 1 )) && noun="tick"
  stale_banner="⚠️ *stale — GitHub API unreachable; the status below is NOT current.*"
  stale_banner+=$'\n'"last successful check: ${last_ok:-unknown}  ·  last tick: $ts_human  ·  $fail_streak consecutive failed $noun"
fi

payload=$(jq -nc \
  --arg title "$title" \
  --arg succ "$success_line" \
  --arg ticks "$ticks_line" \
  --arg stale "$stale_banner" \
  --args \
  '{
     text: $title,
     blocks: (
       [{type: "header", text: {type: "plain_text", text: $title, emoji: true}}]
       + (if $stale != "" then [{type: "section", text: {type: "mrkdwn", text: $stale}}] else [] end)
       + (if $ticks != "" then [{type: "context", elements: [{type: "mrkdwn", text: $ticks}]}] else [] end)
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
  [[ -n "$stale_banner" ]] && printf '%s\n' "$stale_banner"
  [[ -n "$success_line" ]] && echo "$success_line"
  for b in "${failure_blocks[@]+"${failure_blocks[@]}"}"; do
    echo "----------------------------------------------------------------"
    printf '%s\n' "$b"
  done
  echo "================================================================"
elif [[ "$slack_mode" == "bot" ]]; then
  # Same status as the standing message → chat.update it (tick appended).
  # Different status / no standing message / update failed → chat.postMessage
  # a fresh one. Either way persist {ts, fingerprint, ticks} under _slack.
  if [[ -n "$msg_ts" ]]; then
    resp=$(printf '%s' "$payload" \
           | jq -c --arg ch "$SLACK_CHANNEL_ID" --arg ts "$msg_ts" '. + {channel: $ch, ts: $ts}' \
           | curl -sS -X POST -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
                  -H 'Content-Type: application/json; charset=utf-8' \
                  --data @- https://slack.com/api/chat.update)
    if [[ "$(jq -r '.ok // false' <<<"$resp")" == "true" ]]; then
      log "same status — updated Slack digest ts=$msg_ts (tick $(jq 'length' <<<"$ticks_json"), ${#payload} chars JSON)"
    else
      log "WARN: chat.update failed ($(jq -r '.error // "unparseable"' <<<"$resp")) — posting fresh digest"
      msg_ts=""
    fi
  fi
  if [[ -z "$msg_ts" ]]; then
    resp=$(printf '%s' "$payload" \
           | jq -c --arg ch "$SLACK_CHANNEL_ID" '. + {channel: $ch}' \
           | curl -sS -X POST -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
                  -H 'Content-Type: application/json; charset=utf-8' \
                  --data @- https://slack.com/api/chat.postMessage)
    if [[ "$(jq -r '.ok // false' <<<"$resp")" == "true" ]]; then
      msg_ts=$(jq -r '.ts' <<<"$resp")
      log "status changed/new — posted Slack digest ts=$msg_ts (${#payload} chars JSON)"
    else
      log "WARN: chat.postMessage failed: $resp"
    fi
  fi
  if [[ -n "$msg_ts" ]]; then
    # last_ok + fail_streak drive the staleness banner; last_tick records the
    # attempt itself, so "when did it last run at all" is answerable from
    # state.json even when Slack was never touched.
    jq --arg ch "$SLACK_CHANNEL_ID" --arg ts "$msg_ts" --arg fp "$fingerprint" \
       --argjson ticks "$ticks_json" \
       --arg ok "$last_ok" --arg lasttick "$ts_human" --argjson streak "$fail_streak" \
       '._slack = {channel: $ch, ts: $ts, fingerprint: $fp, ticks: $ticks,
                   last_ok: $ok, last_tick: $lasttick, fail_streak: $streak}' \
       "$STATE" > "$STATE.tmp" && mv "$STATE.tmp" "$STATE"
  fi
else
  resp=$(printf '%s' "$payload" \
         | curl -sS -X POST -H 'Content-Type: application/json' --data @- "$SLACK_URL")
  if [[ "$resp" == "ok" ]]; then
    log "posted to Slack (${#payload} chars JSON)"
  else
    log "WARN: Slack response: $resp"
  fi
fi

# Feed the duplicate-tick debounce at the top of the script.
if [[ "$DRY_RUN" != "1" ]]; then
  jq --argjson t "$(date +%s)" '._last_tick_end = $t' "$STATE" > "$STATE.tmp" \
    && mv "$STATE.tmp" "$STATE"
fi
