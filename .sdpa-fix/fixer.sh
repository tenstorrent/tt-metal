#!/usr/bin/env bash
# SDPA autofix — turns the SDPA watcher's ❌ runs into DRAFT PRs marked for
# human review. Cron'd hourly at :10 (between the sdpa :00 and conv :30
# watchers). See README.md for the design; config.sh for knobs.
#
#   FIX_MODE=dryrun|live   (default from config.sh)
#   ONLY=<workflow.yaml>   restrict this tick to one pipeline (manual runs)
#   NO_FIX=1               triage + ledger only, skip the fix stage
#   FORCE_SIG=<sig>        run the fix agent on this one signature (any kind)
#   DECISIONS_ONLY=1       only apply decisions clicked in Slack (set by decide.py)
set -euo pipefail

FIX_HOME="$HOME/.sdpa-fix"
source "$FIX_HOME/config.sh"
FIXLIB="python3 $FIX_HOME/fixlib.py"
WATCH_STATE="$HOME/.sdpa-watch/state.json"
AGENT_ERR="$FIX_HOME/agent_errors.log"
LOG_DIR="$FIX_HOME/logs"
mkdir -p "$LOG_DIR" "$FIX_HOME/proposals" "$FIX_HOME/runs"

if [[ ! -t 1 ]]; then
  exec >>"$LOG_DIR/$(date -u +%F).log" 2>&1
  ln -sfn "$(date -u +%F).log" "$LOG_DIR/today.log" 2>/dev/null || true
fi
ts()  { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(ts)] $*" >&2; }

exec 201>"$FIX_HOME/.fix.lock"
if ! flock -n 201; then
  log "another fixer.sh holds the lock — skipping"
  exit 0
fi
log "==== fixer tick (mode=$FIX_MODE) ===="

# Reuse the watcher's own implementations instead of copies, so a fix to one
# (auth refresh, log extraction) reaches both. Extracted by function name.
# shellcheck disable=SC1090
source <(sed -n '/^refresh_oauth_credential() {/,/^}/p; /^fetch_failure_logs() {/,/^}/p; /^job_log_excerpt() {/,/^}/p' \
         "$HOME/.sdpa-watch/watch.sh")
declare -F refresh_oauth_credential >/dev/null && declare -F fetch_failure_logs >/dev/null \
  && declare -F job_log_excerpt >/dev/null \
  || { log "FATAL: could not import helpers from watch.sh"; exit 1; }

# ---------- auth (same two modes as the watcher) ----------
unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN
if [[ -s "${OAUTH_TOKEN_FILE:-}" ]]; then
  CLAUDE_CODE_OAUTH_TOKEN="$(tr -d '[:space:]' < "$OAUTH_TOKEN_FILE")"; export CLAUDE_CODE_OAUTH_TOKEN
else
  # The refresh token ROTATES: two processes refreshing at once can strand one
  # of them with a dead token. Refresh only while holding the watcher's lock.
  ( flock -w 900 9 || exit 0; refresh_oauth_credential || true ) 9>"$HOME/.sdpa-watch/.watch.lock"
fi
for m in "$TRIAGE_MODEL" "$FIX_MODEL"; do
  if ! printf 'reply with the single word OK' | claude --model "$m" -p >/dev/null 2>>"$AGENT_ERR"; then
    log "FATAL: auth/model preflight failed for $m — see $AGENT_ERR"
    exit 1
  fi
done

SLACK_BOT_TOKEN=""
[[ -f "${SLACK_BOT_TOKEN_FILE:-}" ]] && SLACK_BOT_TOKEN="$(tr -d '[:space:]' < "$SLACK_BOT_TOKEN_FILE")"
# Every fixer message is a threaded reply under the watcher's CURRENT digest
# message (its ts lives in the watcher's state.json), so PR activity sits next
# to the failure it is about instead of cluttering the channel. Falls back to
# a top-level post if the watcher has not posted a digest yet.
# Make GitHub references clickable in Slack mrkdwn: full PR URLs become
# <url|#N>, bare #N (4-6 digits) become PR links, and `commit:<sha>` tokens
# become commit links showing the first 10 chars. CI runs are written as
# <run-url|run:N> by the callers: run numbers must never pass through the
# bare-#N rule (L2's run numbers are 5 digits, like PR numbers).
linkify() {
  local base="https://github.com/$REPO"
  sed -E \
    -e "s#$base/pull/([0-9]+)#<$base/pull/\\1|PR_\\1>#g" \
    -e "s#(^|[^|/&A-Za-z0-9_])\\#([0-9]{4,6})\\b#\\1<$base/pull/\\2|PR_\\2>#g" \
    -e "s#commit:([0-9a-f]{10})([0-9a-f]*)#<$base/commit/\\1\\2|\\1>#g" \
    -e "s#\\|PR_([0-9]+)>#|\\#\\1>#g" \
    -e "s#run:([0-9]+)#run \\#\\1#g"
}

slack() {
  [[ "$FIX_SLACK" == "1" && -n "$SLACK_BOT_TOKEN" && -n "${SLACK_CHANNEL_ID:-}" ]] || { log "  (slack) $1"; return 0; }
  local resp thread
  thread=$(jq -r --arg ch "$SLACK_CHANNEL_ID" 'if (._slack.channel // "") == $ch then (._slack.ts // "") else "" end' \
             "$WATCH_STATE" 2>/dev/null || true)
  resp=$(jq -nc --arg ch "$SLACK_CHANNEL_ID" --arg t "$(linkify <<<"$1")" --arg th "$thread" \
           '{channel:$ch, text:$t, unfurl_links:false} + (if $th != "" then {thread_ts:$th} else {} end)' \
         | curl -sS -X POST -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
                -H 'Content-Type: application/json; charset=utf-8' --data @- https://slack.com/api/chat.postMessage)
  SLACK_LAST_TS=""
  if [[ "$(jq -r '.ok // false' <<<"$resp")" == "true" ]]; then
    # Keep every message's ts: the bot has no history scope, so this is the
    # only way to chat.update a message later.
    SLACK_LAST_TS=$(jq -r .ts <<<"$resp")
    jq -nc --arg ts "$SLACK_LAST_TS" --arg th "$thread" --arg t "${1:0:160}" --arg at "$(ts)" \
       '{at:$at, ts:$ts, thread:$th, text:$t}' >> "$FIX_HOME/slack_posts.jsonl"
  else
    log "  WARN: slack post failed: $resp"
  fi
}

# Replace the text of an earlier bot message in place (no new notification).
slack_update() {
  local ts="$1" text="$2" resp
  [[ "$FIX_SLACK" == "1" && -n "$SLACK_BOT_TOKEN" && -n "${SLACK_CHANNEL_ID:-}" ]] || { log "  (slack edit $ts) $text"; return 0; }
  resp=$(jq -nc --arg ch "$SLACK_CHANNEL_ID" --arg ts "$ts" --arg t "$(linkify <<<"$text")" \
           '{channel:$ch, ts:$ts, text:$t, blocks:[{type:"section", text:{type:"mrkdwn", text:$t}}]}' \
         | curl -sS -X POST -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
                -H 'Content-Type: application/json; charset=utf-8' --data @- https://slack.com/api/chat.update)
  [[ "$(jq -r '.ok // false' <<<"$resp")" == "true" ]] && return 0
  log "  WARN: slack edit of $ts failed: $(jq -r '.error // "?"' <<<"$resp")"
  return 1
}

# ONE Slack message per fix. The first post about these signatures stores its
# ts in the ledger; every later status (opened, CI result, merged, verified)
# edits that message instead of adding a reply. Falls back to a new reply when
# there is nothing to edit or the edit fails.
slack_sig() {
  local text="$1"; shift
  local mts recs foot
  recs=$($FIXLIB get "$@")
  mts=$(jq -r '[.[] | .slack_ts // empty] | first // ""' <<<"$recs")
  # A human decision stays visible through every later edit of the message.
  foot=$(jq -r '[.[] | select(.decision_choice != null)] | first
                | if . == null then "" else
                    (.decision_choice.key as $k
                     | "_decision: \(if $k == "REJECT" then "Reject" else ([.decision.options[] | select(.key == $k) | "\(.key) · \(.label)"] | first) end)"
                       + " — chosen by <@\(.decision_choice.by)> at \(.decision_choice.at[0:16] | sub("T"; " ")) UTC_")
                  end' <<<"$recs")
  [[ -n "$foot" ]] && text+=$'\n'"$foot"
  foot=$(jq -r '[.[] | .decision_history // [] | .[] | select(.note != null)
                 | "_✍️ <@\(.note.by)>: “\(.note.text[0:200])”_"] | unique | join("\n")' <<<"$recs")
  [[ -n "$foot" ]] && text+=$'\n'"$foot"
  if [[ -n "$mts" ]] && slack_update "$mts" "$text"; then return 0; fi
  slack "$text"
  [[ -n "${SLACK_LAST_TS:-}" ]] && $FIXLIB mark --state keep \
    --extra "$(jq -nc --arg t "$SLACK_LAST_TS" '{slack_ts:$t}')" "$@"
  return 0
}

run_url() { echo "https://github.com/$REPO/actions/runs/$1"; }

# Who owns a PR, for "by @author" labels on PRs that are not ours.
pr_author() { gh pr view "$1" -R "$REPO" --json author --jq '.author.login' 2>/dev/null || true; }

# Message wording convention: a PR that needs one of US to act (our autofix
# draft to review, a dry-run proposal to look at) is labelled
# *autofix draft #N — needs your review* in bold; a PR by someone else is
# "#N by @author" in plain text; our merged fix is plain "autofix #N merged".

run_jobs_json() {  # all jobs of a run as [{name, conclusion}]; 3 tries (GitHub 5xx are common)
  local i out
  for i in 1 2 3; do
    if out=$(gh api "repos/$REPO/actions/runs/$1/jobs?per_page=100" --paginate \
               --jq '[.jobs[] | {id, name, conclusion}]' 2>>"$AGENT_ERR" | jq -s 'add // []'); then
      printf '%s' "$out"; return 0
    fi
    sleep $((i * 10))
  done
  return 1
}

# GitHub's run list serves a stale index for branch=/event= filtered queries
# (2026-10: L2's "latest" main run came back as September's #9871 for days,
# freezing the digest behind the stale-page guard). Adding a created>= window
# routes the query to a fresh index. 10 days covers every watched schedule.
RECENT="&created=>$(date -u -d '-10 days' +%F)"

# ---------------------------------------------------------------------------
# Decisions: when a fix is a human call, the bot opens NOTHING. It posts one
# Slack message with a button per option (+ Reject) and waits; decide.py
# (Socket Mode listener) records the click in the ledger and runs
# DECISIONS_ONLY=1 fixer.sh, which applies the choice right away.
# ---------------------------------------------------------------------------
slack_decision() {  # $1 sig(s) as space list, $2 header text, $3 decision json
  local sigs_s="$1" head="$2" dec="$3" thread resp blocks first rec old oldth hist
  first="${sigs_s%% *}"
  rec=$($FIXLIB get "$first" | jq -c --arg s "$first" '.[$s]')
  old=$(jq -r '.slack_ts // ""' <<<"$rec"); oldth=$(jq -r '.decision_thread // ""' <<<"$rec")
  # Earlier rounds: what was written in "Other…" before this poll.
  hist=$(jq -r '[.decision_history // [] | to_entries[] | select(.value.note != null)
                 | "round \(.key + 1): <@\(.value.note.by)> wrote “\(.value.note.text[0:300])”"] | join("\n")' <<<"$rec")
  [[ "$FIX_SLACK" == "1" && -n "$SLACK_BOT_TOKEN" && -n "${SLACK_CHANNEL_ID:-}" ]] || { log "  (slack decision) $head"; return 0; }
  thread=$(jq -r --arg ch "$SLACK_CHANNEL_ID" 'if (._slack.channel // "") == $ch then (._slack.ts // "") else "" end' \
             "$WATCH_STATE" 2>/dev/null || true)
  blocks=$(jq -nc --arg t "$(linkify <<<"$head")" --arg s "$first" --argjson d "$dec" --arg h "$hist" '
    [ {type: "section", text: {type: "mrkdwn", text: $t}},
      {type: "context", elements: [{type: "mrkdwn", text: ($d.options | map("*\(.key)* · \(.summary)") | join("\n"))}]}]
    + (if $h != "" then [{type: "context", elements: [{type: "mrkdwn", text: ("✍️ " + $h)}]}] else [] end)
    + [
      {type: "actions", block_id: "autofix_decide",
       elements: (($d.options | map({type: "button", action_id: ("autofix_decide_" + .key),
                                     text: {type: "plain_text", text: ("\(.key) · \(.label)" + (if .key == $d.recommended then " ★" else "" end))},
                                     value: ({sig: $s, key: .key} | tostring)}
                                    + (if .key == $d.recommended then {style: "primary"} else {} end)))
                  + [{type: "button", action_id: "autofix_decide_OTHER",
                      text: {type: "plain_text", text: "✍️ Other…"}, value: ({sig: $s, key: "OTHER"} | tostring)},
                     {type: "button", action_id: "autofix_decide_REJECT", style: "danger",
                      text: {type: "plain_text", text: "Reject"}, value: ({sig: $s, key: "REJECT"} | tostring)}])} ]')
  # Same thread as before: replace the poll in place (next round of a loop).
  if [[ -n "$old" && "$oldth" == "$thread" ]]; then
    resp=$(jq -nc --arg ch "$SLACK_CHANNEL_ID" --arg ts "$old" --arg t "$(linkify <<<"$head")" --argjson b "$blocks" \
             '{channel:$ch, ts:$ts, text:$t, blocks:$b}' \
           | curl -sS -X POST -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
                  -H 'Content-Type: application/json; charset=utf-8' --data @- https://slack.com/api/chat.update)
    [[ "$(jq -r '.ok // false' <<<"$resp")" == "true" ]] && return 0
    log "  WARN: decision edit failed, posting anew: $(jq -r '.error // "?"' <<<"$resp")"
  fi
  resp=$(jq -nc --arg ch "$SLACK_CHANNEL_ID" --arg t "$(linkify <<<"$head")" --arg th "$thread" --argjson b "$blocks" \
           '{channel:$ch, text:$t, blocks:$b, unfurl_links:false} + (if $th != "" then {thread_ts:$th} else {} end)' \
         | curl -sS -X POST -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
                -H 'Content-Type: application/json; charset=utf-8' --data @- https://slack.com/api/chat.postMessage)
  if [[ "$(jq -r '.ok // false' <<<"$resp")" == "true" ]]; then
    # shellcheck disable=SC2086
    $FIXLIB mark --state keep --extra "$(jq -nc --arg t "$(jq -r .ts <<<"$resp")" --arg th "$thread" '{slack_ts:$t, decision_thread:$th}')" $sigs_s
  else
    log "  WARN: decision post failed: $resp"
  fi
}

# An unanswered decision must sit under the CURRENT digest: when the watcher
# posts a new digest message (status changed), move the question there —
# delete the old message, post it again with the same buttons.
refresh_decisions() {
  local row sig cur old
  cur=$(jq -r --arg ch "${SLACK_CHANNEL_ID:-}" 'if (._slack.channel // "") == $ch then (._slack.ts // "") else "" end' \
          "$WATCH_STATE" 2>/dev/null || true)
  [[ -n "$cur" && "$FIX_SLACK" == "1" && -n "$SLACK_BOT_TOKEN" ]] || return 0
  while IFS= read -r row; do
    [[ -z "$row" ]] && continue
    sig=$(jq -r .sig <<<"$row"); old=$(jq -r '.r.slack_ts // ""' <<<"$row")
    if [[ -n "$old" ]]; then
      jq -nc --arg ch "$SLACK_CHANNEL_ID" --arg ts "$old" '{channel:$ch, ts:$ts}' \
        | curl -sS -X POST -H "Authorization: Bearer $SLACK_BOT_TOKEN" -H 'Content-Type: application/json; charset=utf-8' \
               --data @- https://slack.com/api/chat.delete >/dev/null || true
    fi
    log "  moving open decision for $sig under the current digest"
    slack_decision "$sig" "❓ *autofix — needs your decision*: \`$(jq -r '.r.test | sub(".*::"; "")' <<<"$row")\` ($(jq -r .r.workflow <<<"$row"))
$(jq -r .r.decision.question <<<"$row")" "$(jq -c .r.decision <<<"$row")"
  done < <(jq -c --arg cur "$cur" '.sigs | to_entries[]
             | select(.value.state == "awaiting_decision" and .value.decision_choice == null
                      and (.value.decision_thread // "") != $cur) | {sig: .key, r: .value}' "$FIX_HOME/ledger.json")
}

apply_decisions() {
  local row sig key by short wf kind label cpr who
  while IFS= read -r row; do
    [[ -z "$row" ]] && continue
    sig=$(jq -r .sig <<<"$row"); key=$(jq -r .r.decision_choice.key <<<"$row"); by=$(jq -r .r.decision_choice.by <<<"$row")
    short=$(jq -r '.r.test | sub(".*::"; "")' <<<"$row"); wf=$(jq -r .r.workflow <<<"$row")
    if [[ "$key" == "REJECT" ]]; then
      $FIXLIB mark --state rejected "$sig"
      log "  decision for $short: rejected by $by"
      slack_sig "⚪ rejected: no change for \`$short\` ($wf)" "$sig"
      continue
    fi
    kind=$(jq -r --arg k "$key" '.r.decision.options[] | select(.key == $k) | .kind' <<<"$row")
    label=$(jq -r --arg k "$key" '.r.decision.options[] | select(.key == $k) | .label' <<<"$row")
    case "$kind" in
      patch)
        $FIXLIB mark --state decided "$sig"
        log "  decision for $short: $key ($label) by $by — fix stage implements it now"
        slack_sig "🛠 preparing the autofix draft PR for \`$short\` ($wf)…" "$sig" ;;
      *) log "  WARN: unknown decision $key for $sig" ;;
    esac
  done < <(jq -c '.sigs | to_entries[] | select(.value.state == "awaiting_decision" and .value.decision_choice != null) | {sig: .key, r: .value}' "$FIX_HOME/ledger.json")
}

# ======================================================================
# Phase A — follow up on open draft PRs (live mode)
# ======================================================================
followup() {
  local open
  open=$(jq -c '[.sigs | to_entries[] | select((.value.state=="pr_open" or .value.state=="ci_passed" or .value.state=="ci_failed") and .value.pr != null)
                 | {sig: .key, pr: .value.pr, st: .value.state, wf: .value.workflow, dispatched: (.value.dispatched // []),
                    title: ((.value.verdict_title // "") | sub("^\\[autofix[^]]*\\] *"; "")), test: (.value.test | sub(".*::"; ""))}]
                | group_by(.pr.url)[] | {pr: .[0].pr, st: .[0].st, wf: .[0].wf, dispatched: .[0].dispatched,
                                         title: .[0].title, tests: (map(.test) | unique | join(", ")), sigs: map(.sig)}' \
         "$FIX_HOME/ledger.json" 2>/dev/null || true)
  [[ -z "$open" ]] && return 0
  while IFS= read -r g; do
    local url num sigs state all_done any_fail lines
    url=$(jq -r '.pr.url' <<<"$g"); num=$(jq -r '.pr.number' <<<"$g")
    mapfile -t sigs < <(jq -r '.sigs[]' <<<"$g")
    state=$(gh pr view "$url" --json state,isDraft --jq '.state' 2>/dev/null || echo "")
    case "$state" in
      MERGED)
        # Merged is NOT done: the fix only counts once a later run of the same
        # pipeline that contains this merge commit goes green (fixlib update
        # turns merged → verified, or back to tracking if still red).
        local msha
        msha=$(gh pr view "$url" --json mergeCommit --jq '.mergeCommit.oid' 2>/dev/null || echo "")
        $FIXLIB mark --state merged --extra "$(jq -nc --arg f "$msha" '{fix_sha:$f}')" "${sigs[@]}"
        slack_sig "$EMOJI_PR_MERGED autofix #$num merged: $(jq -r .title <<<"$g")
• \`$(jq -r .tests <<<"$g")\` ($(jq -r .wf <<<"$g")) · merge commit:$msha
• waiting for the next run that contains it to confirm it is green" "${sigs[@]}"
        continue ;;
      CLOSED) $FIXLIB mark --state rejected "${sigs[@]}"; log "  PR $url closed by a human — signature(s) rejected"; continue ;;
    esac
    # Dispatched CI already reported on: only the merge/close check above applies.
    [[ "$(jq -r .st <<<"$g")" == "pr_open" ]] || continue
    all_done=1; any_fail=0; lines=""
    local disp new_disp="[]"
    disp=$(jq -c '.dispatched' <<<"$g")
    while IFS= read -r d; do
      [[ -z "$d" ]] && continue
      local rid concl
      rid=$(jq -r '.run_id // empty' <<<"$d")
      if [[ -z "$rid" ]]; then new_disp=$(jq -c --argjson d "$d" '. + [$d]' <<<"$new_disp"); continue; fi
      concl=$(gh api "repos/$REPO/actions/runs/$rid" --jq '.status + "/" + (.conclusion // "")' 2>/dev/null || echo "?/")
      [[ "$concl" == completed/* ]] || all_done=0
      [[ "$concl" == completed/success ]] || { [[ "$concl" == completed/* ]] && any_fail=1; }
      lines+="• \`$(jq -r .workflow <<<"$d")\` → ${concl#completed/} $(run_url "$rid")"$'\n'
      new_disp=$(jq -c --argjson d "$d" --arg c "$concl" '. + [$d + {conclusion: $c}]' <<<"$new_disp")
    done < <(jq -c '.[]' <<<"$disp")
    (( all_done )) || continue
    local head
    head=$(gh pr view "$url" --json headRefOid --jq .headRefOid)
    if (( any_fail )); then
      gh api "repos/$REPO/statuses/$head" -f state=failure -f context="$CI_STATUS_CONTEXT" \
         -f description="Targeted CI failed — the fix did not hold" >/dev/null
      gh pr comment "$url" --body "❌ **autofix:** the targeted CI legs finished and at least one failed. The fix did not hold; needs a human.
$lines" >/dev/null
      $FIXLIB mark --state ci_failed --extra "{\"dispatched\": $new_disp}" "${sigs[@]}"
      slack_sig "$EMOJI_PR_OPENED *autofix draft #$num — targeted CI ❌, needs your look*: $(jq -r .title <<<"$g")
• \`$(jq -r .tests <<<"$g")\` ($(jq -r .wf <<<"$g"))" "${sigs[@]}"
    else
      gh api "repos/$REPO/statuses/$head" -f state=success -f context="$CI_STATUS_CONTEXT" \
         -f description="Targeted CI passed — still needs human review" >/dev/null
      gh pr comment "$url" --body "✅ **autofix:** all targeted CI legs passed. Still a draft: needs a human to review the diff and mark it ready.
$lines" >/dev/null
      $FIXLIB mark --state ci_passed --extra "{\"dispatched\": $new_disp}" "${sigs[@]}"
      slack_sig "$EMOJI_PR_OPENED *autofix draft #$num — targeted CI ✅, needs your review*: $(jq -r .title <<<"$g")
• \`$(jq -r .tests <<<"$g")\` ($(jq -r .wf <<<"$g"))" "${sigs[@]}"
    fi
  done <<<"$open"
}
# Every mode: dryrun only stops the bot from OPENING PRs. A PR that already
# exists (e.g. from a manual `sdpa-fix` live run) must still be followed to
# merged / closed, or the ledger and digest keep calling it an open draft.
followup
apply_decisions
refresh_decisions

# ======================================================================
# Phase B — triage every newly analyzed run of every watched pipeline
# ======================================================================
handle_green() {  # newly_green JSON from fixlib update
  local ng="$1"
  while IFS= read -r x; do
    [[ -z "$x" ]] && continue
    local prurl test
    prurl=$(jq -r '.pr.url // empty' <<<"$x"); test=$(jq -r '.test' <<<"$x")
    log "  green on main again: $test"
    if [[ -n "$prurl" && "$FIX_MODE" == "live" ]]; then
      local st
      st=$(gh pr view "$prurl" --json state --jq .state 2>/dev/null || echo "")
      if [[ "$st" == "OPEN" ]]; then
        gh pr comment "$prurl" --body "ℹ️ **autofix:** \`$test\` is passing on main again ($(jq -r .run.url <<<"$x")) without this PR. Closing as no longer needed; reopen if you still want the change." >/dev/null
        gh pr close "$prurl" >/dev/null
        slack_sig "⚪ closed $prurl: \`$test\` is green on main again without it" "$(jq -r .sig <<<"$x")"
      fi
    fi
  done < <(jq -c '.newly_green[]' <<<"$ng")
}

# A merged autofix PR or an upstream fix (fixed_upstream) is only "done" once a
# run that CONTAINS the fix commit finishes: green → verified, red → reopened.
handle_fix_events() {
  local ng="$1"
  while IFS= read -r ev; do
    [[ -z "$ev" ]] && continue
    local typ test pr fsha rnum rurl who
    typ=$(jq -r .type <<<"$ev"); test=$(jq -r '.test | sub(".*::"; "")' <<<"$ev")
    pr=$(jq -r '.pr.url // empty' <<<"$ev"); fsha=$(jq -r '.fix_sha // ""' <<<"$ev")
    rnum=$(jq -r .run.number <<<"$ev"); rurl=$(jq -r .run.url <<<"$ev")
    who="${pr:-commit:$fsha}"
    if [[ "$typ" == "verified" ]]; then
      log "  verified green after fix: $test"
      slack_sig "✅ *verified*: \`$test\` passed in <$rurl|run:$rnum>, which contains the fix $who. Nothing more to add." "$(jq -r .sig <<<"$ev")"
    else
      log "  STILL FAILING after fix: $test"
      slack_sig "❌ *still failing after fix*: \`$test\` failed in <$rurl|run:$rnum> even though it contains $who. Back in the autofix queue." "$(jq -r .sig <<<"$ev")"
    fi
  done < <(jq -c '.events[]?' <<<"$ng")
}

for entry in "${PIPELINES[@]}"; do
  [[ "${DECISIONS_ONLY:-0}" == "1" ]] && break
  IFS='|' read -r workflow display test_hint job_pattern <<<"$entry"
  [[ -n "${ONLY:-}" && "$ONLY" != "$workflow" ]] && continue
  st=$(jq -c --arg w "$workflow" '.[$w] // empty' "$WATCH_STATE")
  [[ -z "$st" ]] && continue
  run_id=$(jq -r '.run_id' <<<"$st"); run_number=$(jq -r '.run_number' <<<"$st")
  sha=$(jq -r '.sha' <<<"$st"); summary=$(jq -r '.summary' <<<"$st")
  url=$(run_url "$run_id")

  # Per-job triage: the watcher records the FINISHED in-scope jobs of the run it
  # reports (possibly still in progress). Each job is triaged once, by job id,
  # so a slow leg (Blaze SC4) no longer holds back the ones that finished.
  finished=$(jq -c '.jobs // empty' <<<"$st")
  if [[ -z "$finished" ]]; then
    # State written by the pre-per-job watcher: list the run's jobs ourselves.
    if ! finished=$(run_jobs_json "$run_id" | jq -c --arg p "$job_pattern" \
          '[.[] | select(.conclusion != null) | select($p == "" or (.name | test($p; "i")))]'); then
      log "  GitHub API failed listing jobs for $workflow — skipping this tick"; continue
    fi
  fi
  seen=$($FIXLIB triaged --workflow "$workflow" --run-id "$run_id")
  [[ "$seen" == '"ALL"' ]] && continue
  new=$(jq -c --argjson s "$seen" '[.[] | select((.id | tostring) as $i | ($s | index($i)) | not)]' <<<"$finished")
  [[ "$(jq length <<<"$new")" == "0" ]] && continue
  new_failed=$(jq -c '[.[] | select(.conclusion == "failure")]' <<<"$new")
  job_ids=$(jq -r '[.[].id | tostring] | join(",")' <<<"$new")
  log "triage: $display run #$run_number — $(jq length <<<"$new") newly finished in-scope job(s), $(jq length <<<"$new_failed") failed"

  tri="$FIX_HOME/runs/$run_id.$(date -u +%H%M%S).triage.json"
  jobs="$FIX_HOME/runs/$run_id.$(date -u +%H%M%S).jobs.json"
  jq '[.[] | {name, conclusion}]' <<<"$new" > "$jobs"
  if [[ "$(jq length <<<"$new_failed")" != "0" ]]; then
    logs=""
    while IFS=$'\t' read -r jid jname; do
      logs+="=== JOB: $jname ===
$(job_log_excerpt "$jid")

"
    done < <(jq -r '.[] | "\(.id)\t\(.name)"' <<<"$new_failed")
    # Appended per batch: Phase C reads the latest failing run's excerpts here.
    printf '%s' "$logs" >> "$FIX_HOME/runs/$run_id.logs.txt"
    prompt="$(cat "$FIX_HOME/prompts/triage.txt")

# Context
Pipeline: $display ($workflow)
Run: #$run_number  sha=$sha  $url$( [[ "$(jq -r '.partial == true' <<<"$st")" == "true" ]] && echo "  (run still in progress; only these finished jobs are given)")
Test-focus hint (in-scope rules): $test_hint

Watcher summary of this run:
$summary

Failing job log excerpts (only jobs not triaged before):
$(printf '%s' "$logs" | tail -c 80000)"
    set +e
    out=$(cd "$TT_METAL_DIR" && timeout "$TRIAGE_TIMEOUT_SEC" claude --model "$TRIAGE_MODEL" -p \
            --output-format json --json-schema "$(cat "$FIX_HOME/schemas/triage.json")" \
            --allowedTools "Read" "Grep" "Glob" "Bash(git log:*)" "Bash(git show:*)" "Bash(git blame:*)" \
            <<<"$prompt" 2>>"$AGENT_ERR")
    rc=$?
    set -e
    if [[ $rc -ne 0 ]] || ! jq -e '.structured_output.failures' <<<"$out" >/dev/null 2>&1; then
      log "  triage agent failed (rc=$rc) — will retry next tick"
      continue
    fi
    jq '.structured_output' <<<"$out" > "$tri"
    log "  $(jq -r '.failures | length' "$tri") in-scope failure(s): $(jq -r '[.failures[] | .kind] | group_by(.) | map("\(.[0])×\(length)") | join(", ")' "$tri")"
  else
    echo '{"failures":[]}' > "$tri"
  fi
  ng=$($FIXLIB update --workflow "$workflow" --run-id "$run_id" --run-number "$run_number" \
         --sha "$sha" --url "$url" --triage "$tri" --jobs "$jobs" --job-ids "$job_ids")
  handle_green "$ng"
  handle_fix_events "$ng"
done

# ======================================================================
# Phase B2 — find fixes made by people, for EVERY still-failing signature
# (any triage kind): commits on main since the first failure that touch the
# failing test / the files in the error, and open PRs that mention them. A
# judge call confirms each candidate once (cached in the ledger). Merged fix →
# fixed_upstream (verified by the next run that contains it); open fix →
# fix_pending (the bot does not draft its own fix; merge/close is followed).
# ======================================================================
judge_candidate() {  # $1 failure json, $2 candidate text → verdict JSON (or nothing)
  local prompt out
  prompt="$(cat "$FIX_HOME/prompts/judge.txt")

# Failure
$(jq -r '"test: \(.test)\njob: \(.job) (\(.workflow))\nerror: \(.summary)"' <<<"$1")

# Candidate
$2"
  out=$(cd "$TT_METAL_DIR" && timeout 300 claude --model "$TRIAGE_MODEL" -p --output-format json \
          --json-schema "$(cat "$FIX_HOME/schemas/judge.json")" --allowedTools "Read" "Grep" \
          <<<"$prompt" 2>>"$AGENT_ERR") || return 0
  jq -c '.structured_output // empty' <<<"$out"
}

# Sets PR_TEXT (candidate text) PR_STATE PR_MERGE_SHA PR_TITLE PR_HEAD PR_URL.
# Call it directly, never inside $( ): a subshell would drop the variables.
PR_TEXT=""; PR_STATE=""; PR_MERGE_SHA=""; PR_TITLE=""; PR_HEAD=""; PR_URL=""
pr_info() {
  local j
  PR_TEXT=""; PR_STATE=""; PR_MERGE_SHA=""; PR_TITLE=""; PR_HEAD=""; PR_URL=""
  j=$(gh pr view "$1" -R "$REPO" --json number,title,state,body,files,mergeCommit,headRefName,url 2>/dev/null) || return 1
  PR_STATE=$(jq -r .state <<<"$j"); PR_MERGE_SHA=$(jq -r '.mergeCommit.oid // ""' <<<"$j")
  PR_TITLE=$(jq -r .title <<<"$j"); PR_HEAD=$(jq -r .headRefName <<<"$j"); PR_URL=$(jq -r .url <<<"$j")
  PR_TEXT=$(printf 'PR #%s (%s): %s\nchanged files: %s\ndescription:\n%s\n\ndiff excerpt:\n%s\n' "$1" "$PR_STATE" "$PR_TITLE" \
    "$(jq -r '[.files[].path] | join(", ")' <<<"$j" | cut -c1-1500)" "$(jq -r '.body // ""' <<<"$j" | cut -c1-1500)" \
    "$(gh pr diff "$1" -R "$REPO" 2>/dev/null | head -c 6000)")
}

scan_human_fixes() {
  local judged=0 f sig state short wf first n st msha
  git -C "$TT_METAL_DIR" fetch -q origin main 2>/dev/null || true
  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    sig=$(jq -r .sig <<<"$f"); state=$(jq -r .state <<<"$f")
    short=$(jq -r '.test | sub(".*::"; "")' <<<"$f"); wf=$(jq -r .workflow <<<"$f")

    # A known open fix PR by a person: follow it to merged / closed.
    if [[ "$state" == "fix_pending" ]]; then
      n=$(jq -r '.fix_pr.number' <<<"$f")
      st=$(gh pr view "$n" -R "$REPO" --json state,mergeCommit --jq '.state + " " + (.mergeCommit.oid // "")' 2>/dev/null || echo "")
      case "$st" in
        MERGED*)
          msha="${st#MERGED }"
          $FIXLIB mark --state fixed_upstream \
            --extra "$(jq -nc --arg f "$msha" --arg r "#$n $(jq -r '.fix_pr.title // ""' <<<"$f")" '{fix_sha:$f, reason:$r}')" "$sig"
          log "  fix PR #$n merged for $short"
          slack_sig "$EMOJI_PR_MERGED fixed on main by #$n by @$(jq -r '.fix_author // "?"' <<<"$f"): \`$short\` ($wf)
• commit:$msha · waiting for the next run that contains it to confirm it is green" "$sig" ;;
        CLOSED*)
          $FIXLIB mark --state tracking "$sig"
          log "  fix PR #$n closed unmerged for $short"
          slack_sig "⚪ fix PR #$n for \`$short\` ($wf) was closed without merging; back in the autofix queue" "$sig" ;;
      esac
      continue
    fi

    # Candidates: "kind<TAB>id<TAB>sha", kind = merged | open.
    local cands=() csha csubj head term id kind text v fixes conf reason seen_ids=" "
    first=$(jq -r .first_sha <<<"$f")
    mapfile -t paths < <(jq -r '.paths[]' <<<"$f")
    if [[ -n "$first" && ${#paths[@]} -gt 0 ]] && git -C "$TT_METAL_DIR" cat-file -e "$first^{commit}" 2>/dev/null; then
      while IFS=$'\t' read -r csha csubj; do
        [[ -z "$csha" ]] && continue
        n=$(grep -oE '\(#[0-9]+\)$' <<<"$csubj" | tr -dc 0-9 || true)
        if [[ -n "$n" ]]; then cands+=("merged"$'\t'"pr:$n"$'\t'"$csha"); else cands+=("merged"$'\t'"commit:$csha"$'\t'"$csha"); fi
      done < <(git -C "$TT_METAL_DIR" log --format='%H%x09%s' "$first..origin/main" -- "${paths[@]}" 2>/dev/null | head -6)
    fi
    while IFS= read -r term; do
      [[ -z "$term" ]] && continue
      while IFS=$'\t' read -r n head; do
        [[ -z "$n" || "$head" == "$BRANCH_PREFIX"/* ]] && continue
        cands+=("open"$'\t'"pr:$n"$'\t')
      done < <(gh pr list -R "$REPO" --state open --search "$term in:title,body" -L 5 \
                 --json number,headRefName --jq '.[] | "\(.number)\t\(.headRefName)"' 2>/dev/null || true)
    done < <(jq -r '.terms[]' <<<"$f")

    local c
    for c in "${cands[@]+"${cands[@]}"}"; do
      IFS=$'\t' read -r kind id csha <<<"$c"
      [[ "$seen_ids" == *" $id "* ]] && continue; seen_ids+="$id "
      [[ "$(jq -r --arg i "$id" '.checked[$i] // empty' <<<"$f")" != "" ]] && continue
      (( judged >= SCAN_MAX_JUDGE_PER_TICK )) && { log "  scan: judge cap reached, rest next tick"; return 0; }
      if [[ "$id" == pr:* ]]; then
        pr_info "${id#pr:}" || continue
        text="$PR_TEXT"
        [[ "$PR_HEAD" == "$BRANCH_PREFIX"/* ]] && continue
        [[ "$PR_STATE" == "MERGED" ]] && { kind=merged; csha="${csha:-$PR_MERGE_SHA}"; }
        [[ "$PR_STATE" == "CLOSED" ]] && continue
      else
        PR_TITLE=$(git -C "$TT_METAL_DIR" log -1 --format=%s "$csha"); PR_URL=""
        text="commit $csha on main: $PR_TITLE
$(git -C "$TT_METAL_DIR" show --stat --format= "$csha" | tail -15)

diff excerpt:
$(git -C "$TT_METAL_DIR" show --format= "$csha" | head -c 6000)"
      fi
      judged=$((judged + 1))
      v=$(judge_candidate "$f" "$text")
      [[ -z "$v" ]] && continue
      $FIXLIB mark --state keep --extra "$(jq -nc --arg i "$id" --argjson v "$v" '{checked: {($i): $v}}')" "$sig"
      fixes=$(jq -r .fixes <<<"$v"); conf=$(jq -r .confidence <<<"$v"); reason=$(jq -r .reason <<<"$v")
      log "  scan $short: $id ($kind) → fixes=$fixes symptom_match=$(jq -r .symptom_match <<<"$v") ($conf) $reason"
      # Only a sure verdict is shown to people: high confidence AND the same
      # failure mode. Anything less stays a cached "no" in the ledger.
      [[ "$fixes" == "true" && "$conf" == "high" && "$(jq -r .symptom_match <<<"$v")" == "true" ]] || continue
      n="${id#pr:}"; [[ "$id" == pr:* ]] || n=""
      local who
      if [[ -n "$n" ]]; then who=$(pr_author "$n"); else who=$(git -C "$TT_METAL_DIR" log -1 --format=%an "$csha"); fi
      if [[ "$kind" == "merged" ]]; then
        $FIXLIB mark --state fixed_upstream \
          --extra "$(jq -nc --arg f "$csha" --arg r "${n:+#$n }$PR_TITLE" --arg a "$who" '{fix_sha:$f, reason:$r, fix_author:$a}')" "$sig"
        slack_sig "$EMOJI_PR_MERGED already fixed on main by ${n:+#$n }by @$who: \`$short\` ($wf)
• commit:$csha · waiting for the next run that contains it to confirm it is green" "$sig"
      else
        $FIXLIB mark --state fix_pending \
          --extra "$(jq -nc --arg n "$n" --arg u "$PR_URL" --arg t "$PR_TITLE" --arg a "$who" '{fix_pr: {number: ($n|tonumber), url: $u, title: $t}, fix_author:$a}')" "$sig"
        slack_sig "$EMOJI_PR_OPENED fix in progress by @$who, open PR #$n: \`$short\` ($wf)
• $PR_TITLE · the bot will not draft its own fix" "$sig"
      fi
      break
    done
  done < <($FIXLIB scan-list | jq -c '.[]')
}
[[ "${DECISIONS_ONLY:-0}" == "1" ]] || scan_human_fixes

[[ "${NO_FIX:-0}" == "1" ]] && { log "NO_FIX=1 — skipping fix stage"; exit 0; }

# ======================================================================
# Phase C — attempt fixes for eligible regression groups
# ======================================================================
elig=$($FIXLIB eligible --mode "$FIX_MODE" --min-streak "$MIN_STREAK" --max-per-day "$MAX_NEW_PER_DAY" --only-sig "${FORCE_SIG:-}")
log "eligible groups: $(jq '.groups | length' <<<"$elig") (daily used $(jq .daily_used <<<"$elig")/$MAX_NEW_PER_DAY)"

prepare_worktree() {
  git -C "$TT_METAL_DIR" fetch -q origin main
  if [[ ! -d "$FIX_WORKTREE/.git" && ! -f "$FIX_WORKTREE/.git" ]]; then
    git -C "$TT_METAL_DIR" worktree add -q --detach "$FIX_WORKTREE" origin/main
  fi
  git -C "$FIX_WORKTREE" checkout -q -f --detach origin/main
  git -C "$FIX_WORKTREE" clean -fdq
}

slugify() { tr 'A-Z' 'a-z' <<<"$1" | sed -E 's/[^a-z0-9]+/-/g; s/^-+|-+$//g' | cut -c1-40; }

attempts=0
while IFS= read -r grp; do
  [[ -z "$grp" ]] && continue
  (( attempts >= MAX_FIX_PER_TICK )) && break
  workflow=$(jq -r .workflow <<<"$grp")
  [[ -n "${ONLY:-}" && "$ONLY" != "$workflow" ]] && continue
  [[ "${DECISIONS_ONLY:-0}" == "1" && "$(jq -r '.forced // false' <<<"$grp")" != "true" ]] && continue
  mapfile -t sigs < <(jq -r '.sigs[]' <<<"$grp")
  recs=$($FIXLIB get "${sigs[@]}")
  # A human chose an option for this failure: implement exactly that, live.
  choice=$(jq -c 'to_entries[0].value | if .state == "decided" then {choice: .decision_choice, decision} else empty end' <<<"$recs")
  # The human wrote an instruction under "Other…": build a NEW poll from it.
  note=$(jq -c 'to_entries[0].value | if .state == "revise" then {note: .decision_note, decision, history: (.decision_history // [])} else empty end' <<<"$recs")
  sig8="${sigs[0]:0:8}"
  first_test=$(jq -r 'to_entries[0].value.test' <<<"$recs")
  log "fix candidate [$workflow] group=$(jq -r .group <<<"$grp") sigs=${sigs[*]}"

  # --- GitHub-side de-dup: our own marker or branch already exists ---
  dup=""
  for s in "${sigs[@]}"; do
    hit=$(gh pr list -R "$REPO" --state all --search "autofixsig$s" --json url,number,state \
            --jq '.[0] // empty' 2>/dev/null || true)
    if [[ -n "$hit" ]]; then dup="$hit"; break; fi
  done
  if [[ -z "$dup" ]] && git -C "$TT_METAL_DIR" ls-remote --exit-code -q origin "refs/heads/$BRANCH_PREFIX/$sig8*" >/dev/null 2>&1; then
    dup='{"url":"(branch exists, no PR)"}'
  fi
  if [[ -n "$dup" ]]; then
    log "  already on GitHub: $dup — adopting, not re-creating"
    $FIXLIB mark --state pr_open --extra "{\"pr\": $dup}" "${sigs[@]}"
    continue
  fi

  # Human PRs that may already address it (informational for agent + body).
  fn=$(sed -E 's/.*:://; s/\[.*//' <<<"$first_test")
  related=$(gh pr list -R "$REPO" --state open --search "$fn in:title,body" -L 5 \
              --json url,title --jq '[.[] | "\(.url) (\(.title))"]' 2>/dev/null || echo '[]')

  # Logs for the group's jobs, from the latest failing run.
  last_run=$(jq -r 'to_entries[0].value.last_seen.id' <<<"$recs")
  logs_file="$FIX_HOME/runs/$last_run.logs.txt"
  logs=$( [[ -f "$logs_file" ]] && tail -c 60000 "$logs_file" || echo "(logs unavailable — open the run URL)")

  # Commit range: run before the first failure .. first failing sha.
  first_num=$(jq -r 'to_entries[0].value.first_seen.number' <<<"$recs")
  first_sha=$(jq -r 'to_entries[0].value.first_seen.sha' <<<"$recs")
  evq=""; [[ -n "${PIPELINE_EVENT[$workflow]:-}" ]] && evq="&event=${PIPELINE_EVENT[$workflow]}"
  good_sha=$(gh api "repos/$REPO/actions/workflows/$workflow/runs?branch=$BRANCH$RECENT&status=completed&per_page=30$evq" \
               --jq "[.workflow_runs[] | select(.run_number < $first_num)] | sort_by(.run_number) | last | .head_sha // empty" \
               2>/dev/null || true)

  prepare_worktree
  pdir="$FIX_HOME/proposals/$(date -u +%Y%m%d-%H%M)-$sig8"
  mkdir -p "$pdir"
  printf '%s' "$recs" | jq . > "$pdir/records.json"
  hint=$(printf '%s\n' "${PIPELINES[@]}" | awk -F'|' -v w="$workflow" '$1==w {print $3}')
  pipe_name=$(printf '%s\n' "${PIPELINES[@]}" | awk -F'|' -v w="$workflow" '$1==w {print $2}')

  prompt="$(cat "$FIX_HOME/prompts/fix.txt")

# Failure context
Pipeline workflow: $workflow
Pipeline in-scope rules: $hint

Failing signatures (triage records):
$(jq '[to_entries[].value | {test, job, sku_jobs, kind, owner, error_key, summary, culprit_sha, culprit_evidence, streak, first_seen, last_seen}]' <<<"$recs")

Last known good run sha (run before the first failure): ${good_sha:-unknown}
First failing run sha: $first_sha
Current worktree HEAD (origin/main): $(git -C "$FIX_WORKTREE" rev-parse HEAD)
Candidate culprit range: git log ${good_sha:-<unknown>}..$first_sha

Possibly related open PRs by humans (if one already fixes this, change nothing and say so in reason_if_not_fixed):
$related

Failing job log excerpts:
$logs"
  if [[ -n "$note" ]]; then
    prompt+="

# Human input on the decision
The previous question was: $(jq -r .decision.question <<<"$note")
Previous options:
$(jq -r '.decision.options[] | "- \(.key): \(.label) — \(.summary)"' <<<"$note")
$(jq -r '.history[] | select(.note != null) | "Earlier the human wrote: \(.note.text)"' <<<"$note")
Now the human wrote: $(jq -r .note.text <<<"$note")

Prepare a NEW decision from this. Do not edit any file. Return decision.needed=true with up to 3 options: option A follows the human's instruction as closely as is allowed; keep earlier options only if still useful. If the instruction asks for something the hard rules forbid, say so in the question and offer the closest allowed options. recommended = the option you would pick now."
  fi
  if [[ -n "$choice" ]]; then
    prompt+="

# Human decision
$(jq -r '.choice.key as $k | .decision.options[] | select(.key == $k) | "The human chose option \(.key): \(.label) — \(.summary). Implement exactly this option."' <<<"$choice")
Original question: $(jq -r .decision.question <<<"$choice")"
  fi

  log "  running fix agent ($FIX_MODEL) in $FIX_WORKTREE"
  set +e
  out=$(cd "$FIX_WORKTREE" && timeout "$FIX_TIMEOUT_SEC" claude --model "$FIX_MODEL" -p \
          --output-format json --json-schema "$(cat "$FIX_HOME/schemas/fix.json")" \
          --permission-mode acceptEdits \
          --allowedTools "Read" "Edit" "Write" "Grep" "Glob" \
            "Bash(git log:*)" "Bash(git show:*)" "Bash(git diff:*)" "Bash(git blame:*)" \
            "Bash(git status:*)" "Bash(git revert --no-commit:*)" \
            "Bash(ls:*)" "Bash(find:*)" "Bash(grep:*)" "Bash(head:*)" "Bash(tail:*)" "Bash(wc:*)" \
          --disallowedTools "Bash(git push:*)" "Bash(git commit:*)" "Bash(pytest:*)" "Bash(python:*)" \
            "Bash(python3:*)" "Bash(*run_safe_pytest*)" "Bash(*tt-smi*)" "Bash(*build_metal*)" "WebFetch" "WebSearch" \
          <<<"$prompt" 2>>"$AGENT_ERR")
  rc=$?
  set -e
  attempts=$((attempts + 1))
  printf '%s' "$out" > "$pdir/agent_result.json"
  if [[ $rc -ne 0 ]] || ! jq -e '.structured_output.fixed != null' <<<"$out" >/dev/null 2>&1; then
    log "  fix agent failed (rc=$rc) — leaving as tracking, will retry"
    $FIXLIB mark --state tracking --attempt --extra "{\"proposal\": \"$pdir\", \"reason\": \"agent error rc=$rc\"}" "${sigs[@]}"
    continue
  fi
  jq '.structured_output' <<<"$out" > "$pdir/verdict.json"
  git -C "$FIX_WORKTREE" add -A
  git -C "$FIX_WORKTREE" diff --cached HEAD > "$pdir/patch.diff"
  v_fixed=$(jq -r .fixed "$pdir/verdict.json")
  v_title=$(jq -r .title "$pdir/verdict.json")

  if [[ -n "$note" ]]; then
    # A revise round never edits or opens anything: it only produces a new poll.
    git -C "$FIX_WORKTREE" reset -q --hard && git -C "$FIX_WORKTREE" clean -fdq
    hist=$(jq -c '.history + [{question: .decision.question, options: .decision.options, note: .note}]' <<<"$note")
    if [[ "$(jq -r '.decision.needed // false' "$pdir/verdict.json")" == "true" ]]; then
      dec=$(jq -c --argjson old "$(jq -c .decision <<<"$note")" '.decision + {culprit_pr: ($old.culprit_pr // ""), proposal: ($old.proposal // "")}' "$pdir/verdict.json")
      extra=""
    else
      dec=$(jq -c .decision <<<"$note")
      extra="
_could not turn that into new options: $(jq -r '.reason_if_not_fixed // .why // "no reason given"' "$pdir/verdict.json")_"
    fi
    $FIXLIB mark --state awaiting_decision --attempt \
      --extra "$(jq -nc --argjson d "$dec" --argjson h "$hist" '{decision: $d, decision_history: $h, decision_note: null}')" "${sigs[@]}"
    log "  new poll after human note (round $(( $(jq length <<<"$hist") + 1 ))): $(jq -r .question <<<"$dec")"
    slack_decision "${sigs[*]}" "❓ *autofix — needs your decision*: \`$first_test\` ($workflow)
$(jq -r .question <<<"$dec")$extra" "$dec"
    continue
  fi

  if [[ -z "$choice" && "$(jq -r '.decision.needed // false' "$pdir/verdict.json")" == "true" ]]; then
    dec=$(jq -c '.decision' "$pdir/verdict.json")
    cpr=$(grep -oE '#[0-9]{4,6}' <<<"$(jq -r '.culprit_title + " " + .why' "$pdir/verdict.json")" | head -1 | tr -d '#' || true)
    [[ -z "$cpr" && -n "$(jq -r .culprit_sha "$pdir/verdict.json")" ]] && \
      cpr=$(git -C "$TT_METAL_DIR" log -1 --format=%s "$(jq -r .culprit_sha "$pdir/verdict.json")" 2>/dev/null | grep -oE '\(#[0-9]+\)$' | tr -dc 0-9 || true)
    dec=$(jq -c --arg c "$cpr" --arg p "$pdir" '. + {culprit_pr: $c, proposal: $p}' <<<"$dec")
    $FIXLIB mark --state awaiting_decision --attempt --extra "$(jq -nc --argjson d "$dec" '{decision: $d, proposal: $d.proposal}')" "${sigs[@]}"
    log "  needs a human decision: $(jq -r .question <<<"$dec")"
    slack_decision "${sigs[*]}" "❓ *autofix — needs your decision*: \`$first_test\` ($workflow)
$(jq -r .question <<<"$dec")" "$dec"
    git -C "$FIX_WORKTREE" reset -q --hard && git -C "$FIX_WORKTREE" clean -fdq
    continue
  fi

  if [[ "$v_fixed" != "true" ]]; then
    upstream=$(jq -r .already_fixed_upstream "$pdir/verdict.json")
    reason=$(jq -r .reason_if_not_fixed "$pdir/verdict.json")
    if [[ -n "$upstream" ]]; then
      # Not a failure of the bot: main already has the fix, the nightly just
      # has not picked it up. Resolves to resolved_on_main on the next green run.
      log "  already fixed upstream: $upstream"
      fsha=$(grep -oE '\b[0-9a-f]{7,40}\b' <<<"$upstream" | head -1 || true)
      $FIXLIB mark --state fixed_upstream --attempt \
        --extra "$(jq -nc --arg p "$pdir" --arg r "$upstream" --arg f "$fsha" '{proposal:$p, reason:$r, fix_sha:$f}')" "${sigs[@]}"
      upr=$(grep -oE '#[0-9]{4,6}' <<<"$upstream" | head -1 | tr -d '#' || true)
      who=$( [[ -n "$upr" ]] && pr_author "$upr" || git -C "$TT_METAL_DIR" log -1 --format=%an "$fsha" 2>/dev/null || true)
      $FIXLIB mark --state keep --extra "$(jq -nc --arg a "$who" '{fix_author:$a}')" "${sigs[@]}"
      slack_sig "$EMOJI_PR_MERGED already fixed on main by ${upr:+#$upr }by @$who: \`$first_test\` ($workflow)
• commit:$fsha · waiting for the next run that contains it to confirm it is green" "${sigs[@]}"
    else
      log "  no fix: $reason"
      $FIXLIB mark --state no_fix --attempt --extra "$(jq -nc --arg p "$pdir" --arg r "$reason" '{proposal:$p, reason:$r}')" "${sigs[@]}"
      slack_sig "🛠 autofix looked at \`$first_test\` ($workflow): *no safe fix*. $reason" "${sigs[@]}"
    fi
    git -C "$FIX_WORKTREE" reset -q --hard && git -C "$FIX_WORKTREE" clean -fdq
    continue
  fi

  guard=$($FIXLIB guard --verdict "$pdir/verdict.json" --diff "$pdir/patch.diff" --max-lines "$MAX_DIFF_LINES")
  printf '%s' "$guard" | jq . > "$pdir/guard.json"
  if [[ "$(jq -r .ok <<<"$guard")" != "true" ]]; then
    reason="guard rejected the diff: $(jq -r '.reasons | join("; ")' <<<"$guard")"
    log "  $reason"
    $FIXLIB mark --state no_fix --attempt --extra "$(jq -nc --arg p "$pdir" --arg r "$reason" '{proposal:$p, reason:$r}')" "${sigs[@]}"
    slack_sig "🛠 autofix drafted a fix for \`$first_test\` but the safety guard rejected it: $(jq -r '.reasons | join("; ")' <<<"$guard"). Proposal: \`$pdir\`" "${sigs[@]}"
    git -C "$FIX_WORKTREE" reset -q --hard && git -C "$FIX_WORKTREE" clean -fdq
    continue
  fi

  jobs_list=$(jq -c '[to_entries[].value.sku_jobs[]] | unique' <<<"$recs")
  $FIXLIB dispatch --workflow "$workflow" --jobs "$jobs_list" --repo "$FIX_WORKTREE" > "$pdir/dispatch.json"
  thr=""; [[ "$(jq -r .threshold_change "$pdir/verdict.json")" == "true" ]] && thr=" [threshold]"
  title="$PR_TITLE_PREFIX$thr $v_title"
  branch="$BRANCH_PREFIX/$sig8-$(slugify "$v_title")"   # known before render: the CI badges link it
  printf '%s\n' "$title" > "$pdir/pr_title.txt"
  render() {
    jq -n --slurpfile v "$pdir/verdict.json" --argjson recs "$recs" --slurpfile d "$pdir/dispatch.json" \
          --argjson rel "$related" --arg repo "$REPO" --arg model "$FIX_MODEL" --arg ci "$CI_STATUS_CONTEXT" \
          --arg pipe "$pipe_name" --arg br "$branch" --argjson ch "${choice:-null}" \
          '{verdict: $v[0], records: [$recs | to_entries[].value], sigs: ($recs | keys), dispatch: $d[0],
            related_prs: $rel, repo: $repo, model: $model, ci_ctx: $ci, pipeline: $pipe, branch: $br,
            decision: (if $ch == null then null else ($ch.choice.key as $k | $ch.decision.options[] | select(.key == $k)) end)}' > "$pdir/meta.json"
    $FIXLIB render --meta "$pdir/meta.json" > "$pdir/pr_body.md"
  }
  render
  files=$(jq -r '.files | join(", ")' <<<"$guard")
  conf=$(jq -r .confidence "$pdir/verdict.json"); appr=$(jq -r .approach "$pdir/verdict.json")

  if [[ "$FIX_MODE" != "live" && -z "$choice" ]]; then
    $FIXLIB mark --state proposed_dryrun --attempt --count-daily \
      --extra "$(jq -nc --arg p "$pdir" --arg t "$title" '{proposal:$p, verdict_title:$t}')" "${sigs[@]}"
    log "  DRY RUN — proposal written to $pdir"
    slack_sig "🛠 *autofix dry-run proposal — take a look*: $title
• failing: \`$first_test\` ($workflow, streak $(jq -r '[.[].streak] | max' <<<"$recs"))
• $appr, confidence $conf, $(jq -r '.added + .deleted' <<<"$guard") lines in $files
• proposal: \`$pdir\`" "${sigs[@]}"
    git -C "$FIX_WORKTREE" reset -q --hard && git -C "$FIX_WORKTREE" clean -fdq
    continue
  fi

  # ---------------- live: branch, commit, push, draft PR, dispatch ----------------
  git -C "$FIX_WORKTREE" checkout -q -b "$branch"
  git -C "$FIX_WORKTREE" commit -q --no-verify -F - <<EOF
$v_title

$(jq -r '.why // .root_cause' "$pdir/verdict.json")

Automated fix for a nightly regression in $workflow
($(jq -r 'to_entries[0].value.last_seen.url' <<<"$recs")). Not run on
device; needs human review.

Co-Authored-By: Claude ($FIX_MODEL) <noreply@anthropic.com>
EOF
  # Use whichever gh is on PATH for credentials: the global helper points at
  # /usr/bin/gh, which this host no longer has, and cron has no GIT_ASKPASS.
  git -C "$FIX_WORKTREE" -c credential.helper= -c 'credential.helper=!gh auth git-credential' \
      push -q -u origin "$branch"
  head_sha=$(git -C "$FIX_WORKTREE" rev-parse HEAD)
  pr_url=$(gh pr create -R "$REPO" --draft --base main --head "$branch" --title "$title" \
             --body-file "$pdir/pr_body.md" --label "$PR_LABELS" --assignee @me)
  pr_num="${pr_url##*/}"
  gh api "repos/$REPO/statuses/$head_sha" -f state=pending -f context="$STATUS_CONTEXT" \
     -f description="Automated fix — needs human review before merge" >/dev/null
  gh api "repos/$REPO/statuses/$head_sha" -f state=pending -f context="$CI_STATUS_CONTEXT" \
     -f description="Targeted CI dispatched" >/dev/null
  # Record the PR NOW: anything below may fail, and a PR the ledger does not
  # know about would only be found again through the marker search.
  $FIXLIB mark --state pr_open --attempt --count-daily \
    --extra "$(jq -nc --arg u "$pr_url" --arg n "$pr_num" --arg b "$branch" --arg p "$pdir" \
               '{pr:{url:$u, number:($n|tonumber), branch:$b}, proposal:$p, dispatched:[]}')" "${sigs[@]}"

  # Dispatch and resolve run ids (gh workflow run does not print one).
  disp_out="[]"
  while IFS= read -r p; do
    wf=$(jq -r .workflow <<<"$p")
    if [[ "$(jq -r '.inputs == null' <<<"$p")" == "true" ]]; then disp_out=$(jq -c --argjson p "$p" '. + [$p]' <<<"$disp_out"); continue; fi
    mapfile -t fargs < <(jq -r '.inputs | to_entries[] | "-f", "\(.key)=\(.value)"' <<<"$p")
    t0=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    if ! gh workflow run "$wf" -R "$REPO" --ref "$branch" "${fargs[@]}" 2>>"$AGENT_ERR"; then
      log "  WARN: dispatch of $wf rejected (inputs?) — PR will say to run it by hand"
      disp_out=$(jq -c --argjson p "$p" '. + [$p + {inputs: null, job: (($p.job // "") + " (dispatch rejected, run by hand)")}]' <<<"$disp_out")
      continue
    fi
    rid=""
    for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do
      sleep 10
      rid=$(gh api "repos/$REPO/actions/workflows/$wf/runs?branch=$(jq -rn --arg b "$branch" '$b|@uri')&event=workflow_dispatch&per_page=10" \
              --jq "[.workflow_runs[] | select(.created_at >= \"$t0\")] | sort_by(.created_at) | last | .id // empty" 2>/dev/null || true)
      [[ -n "$rid" ]] && break
    done
    disp_out=$(jq -c --argjson p "$p" --arg r "$rid" --arg u "${rid:+$(run_url "$rid")}" \
                 '. + [$p + {run_id: (if $r=="" then null else $r end), run_url: (if $u=="" then null else $u end)}]' <<<"$disp_out")
  done < <(jq -c '.[]' "$pdir/dispatch.json")
  printf '%s' "$disp_out" | jq . > "$pdir/dispatch.json"
  $FIXLIB mark --state pr_open --extra "$(jq -nc --argjson d "$disp_out" '{dispatched:$d}')" "${sigs[@]}"
  render
  # REST, not `gh pr edit`: gh 2.63's edit queries the retired Projects
  # (classic) API and fails on every PR.
  gh api -X PATCH "repos/$REPO/pulls/$pr_num" -F "body=@$pdir/pr_body.md" >/dev/null \
    || log "  WARN: could not update the PR body with run links"
  log "  opened draft $pr_url"
  slack_sig "$EMOJI_PR_OPENED *autofix draft #$pr_num — needs your review*: $title
• failing: \`$first_test\` ($workflow)
• $appr, confidence $conf · targeted CI dispatched
$pr_url" "${sigs[@]}"
  git -C "$FIX_WORKTREE" checkout -q -f --detach origin/main
done < <(jq -c '.groups[]' <<<"$elig")

log "==== fixer tick done ===="
