#!/usr/bin/env bash
# SDPA autofix — turns the SDPA watcher's ❌ runs into DRAFT PRs marked for
# human review. Cron'd hourly at :15 (between the sdpa :00 and conv :30
# watchers). See README.md for the design; config.sh for knobs.
#
#   FIX_MODE=dryrun|live   (default from config.sh)
#   ONLY=<workflow.yaml>   restrict this tick to one pipeline (manual runs)
#   NO_FIX=1               triage + ledger only, skip the fix stage
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
source <(sed -n '/^refresh_oauth_credential() {/,/^}/p; /^fetch_failure_logs() {/,/^}/p' \
         "$HOME/.sdpa-watch/watch.sh")
declare -F refresh_oauth_credential >/dev/null && declare -F fetch_failure_logs >/dev/null \
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
slack() {
  [[ "$FIX_SLACK" == "1" && -n "$SLACK_BOT_TOKEN" && -n "${SLACK_CHANNEL_ID:-}" ]] || { log "  (slack) $1"; return 0; }
  local resp thread
  thread=$(jq -r --arg ch "$SLACK_CHANNEL_ID" 'if (._slack.channel // "") == $ch then (._slack.ts // "") else "" end' \
             "$WATCH_STATE" 2>/dev/null || true)
  resp=$(jq -nc --arg ch "$SLACK_CHANNEL_ID" --arg t "$1" --arg th "$thread" \
           '{channel:$ch, text:$t, unfurl_links:false} + (if $th != "" then {thread_ts:$th} else {} end)' \
         | curl -sS -X POST -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
                -H 'Content-Type: application/json; charset=utf-8' --data @- https://slack.com/api/chat.postMessage)
  [[ "$(jq -r '.ok // false' <<<"$resp")" == "true" ]] || log "  WARN: slack post failed: $resp"
}

run_url() { echo "https://github.com/$REPO/actions/runs/$1"; }

run_jobs_json() {  # all jobs of a run as [{name, conclusion}]; 3 tries (GitHub 5xx are common)
  local i out
  for i in 1 2 3; do
    if out=$(gh api "repos/$REPO/actions/runs/$1/jobs?per_page=100" --paginate \
               --jq '[.jobs[] | {name, conclusion}]' 2>>"$AGENT_ERR" | jq -s 'add // []'); then
      printf '%s' "$out"; return 0
    fi
    sleep $((i * 10))
  done
  return 1
}

# ======================================================================
# Phase A — follow up on open draft PRs (live mode)
# ======================================================================
followup() {
  local open
  open=$(jq -c '[.sigs | to_entries[] | select((.value.state=="pr_open" or .value.state=="ci_passed" or .value.state=="ci_failed") and .value.pr != null)
                 | {sig: .key, pr: .value.pr, st: .value.state, wf: .value.workflow, dispatched: (.value.dispatched // [])}]
                | group_by(.pr.url)[] | {pr: .[0].pr, st: .[0].st, wf: .[0].wf, dispatched: .[0].dispatched, sigs: map(.sig)}' \
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
        slack "🟣 *merged* autofix #$num: $url
waiting for the next \`$(jq -r .wf <<<"$g")\` run that contains \`${msha:0:10}\` to confirm it is green"
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
      slack "🛠 ❌ autofix draft #$num: targeted CI failed, needs a human: $url"
    else
      gh api "repos/$REPO/statuses/$head" -f state=success -f context="$CI_STATUS_CONTEXT" \
         -f description="Targeted CI passed — still needs human review" >/dev/null
      gh pr comment "$url" --body "✅ **autofix:** all targeted CI legs passed. Still a draft: needs a human to review the diff and mark it ready.
$lines" >/dev/null
      $FIXLIB mark --state ci_passed --extra "{\"dispatched\": $new_disp}" "${sigs[@]}"
      slack "🛠 ✅ autofix draft #$num: targeted CI passed, ready for your review: $url"
    fi
  done <<<"$open"
}
[[ "$FIX_MODE" == "live" ]] && followup

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
        slack "🛠 autofix closed $prurl: \`$test\` is green on main again without it"
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
    who="${pr:-\`${fsha:0:10}\`}"
    if [[ "$typ" == "verified" ]]; then
      log "  verified green after fix: $test"
      slack "✅ *verified*: \`$test\` passed in run #$rnum ($rurl), which contains the fix $who. Nothing more to add."
    else
      log "  STILL FAILING after fix: $test"
      slack "❌ *still failing after fix*: \`$test\` failed in run #$rnum ($rurl) even though it contains $who. Back in the autofix queue."
    fi
  done < <(jq -c '.events[]?' <<<"$ng")
}

for entry in "${PIPELINES[@]}"; do
  IFS='|' read -r workflow display test_hint job_pattern <<<"$entry"
  [[ -n "${ONLY:-}" && "$ONLY" != "$workflow" ]] && continue
  st=$(jq -c --arg w "$workflow" '.[$w] // empty' "$WATCH_STATE")
  [[ -z "$st" ]] && continue
  run_id=$(jq -r '.run_id' <<<"$st"); run_number=$(jq -r '.run_number' <<<"$st")
  sha=$(jq -r '.sha' <<<"$st"); summary=$(jq -r '.summary' <<<"$st")
  head1=$(head -n1 <<<"$summary")
  [[ "$($FIXLIB triaged --workflow "$workflow")" == "$run_id" ]] && continue
  url=$(run_url "$run_id")
  log "triage: $display run #$run_number"

  tri="$FIX_HOME/runs/$run_id.triage.json"
  jobs="$FIX_HOME/runs/$run_id.jobs.json"
  if [[ "$head1" == *❌* || "$head1" == *✅* ]]; then
    if ! run_jobs_json "$run_id" > "$jobs"; then
      log "  GitHub API failed listing jobs — skipping $workflow this tick"; continue
    fi
  fi
  if [[ "$head1" == *❌* ]]; then
    logs=$(fetch_failure_logs "$run_id")
    printf '%s' "$logs" > "$FIX_HOME/runs/$run_id.logs.txt"
    prompt="$(cat "$FIX_HOME/prompts/triage.txt")

# Context
Pipeline: $display ($workflow)
Run: #$run_number  sha=$sha  $url
Test-focus hint (in-scope rules): $test_hint

Watcher summary of this run:
$summary

Failing job log excerpts:
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
  elif [[ "$head1" == *✅* ]]; then
    echo '{"failures":[]}' > "$tri"
  else
    # ⚠️ tests did not run / 🚫 collateral / 🟡 agent error: no evidence
    # about our tests either way — record as seen, change nothing.
    echo '[]' > "$jobs"; echo '{"failures":[]}' > "$tri"
  fi
  ng=$($FIXLIB update --workflow "$workflow" --run-id "$run_id" --run-number "$run_number" \
         --sha "$sha" --url "$url" --triage "$tri" --jobs "$jobs")
  handle_green "$ng"
  handle_fix_events "$ng"
done

[[ "${NO_FIX:-0}" == "1" ]] && { log "NO_FIX=1 — skipping fix stage"; exit 0; }

# ======================================================================
# Phase C — attempt fixes for eligible regression groups
# ======================================================================
elig=$($FIXLIB eligible --mode "$FIX_MODE" --min-streak "$MIN_STREAK" --max-per-day "$MAX_NEW_PER_DAY")
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
  mapfile -t sigs < <(jq -r '.sigs[]' <<<"$grp")
  recs=$($FIXLIB get "${sigs[@]}")
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
  good_sha=$(gh api "repos/$REPO/actions/workflows/$workflow/runs?branch=$BRANCH&status=completed&per_page=30$evq" \
               --jq "[.workflow_runs[] | select(.run_number < $first_num)] | sort_by(.run_number) | last | .head_sha // empty" \
               2>/dev/null || true)

  prepare_worktree
  pdir="$FIX_HOME/proposals/$(date -u +%Y%m%d-%H%M)-$sig8"
  mkdir -p "$pdir"
  printf '%s' "$recs" | jq . > "$pdir/records.json"
  hint=$(printf '%s\n' "${PIPELINES[@]}" | awk -F'|' -v w="$workflow" '$1==w {print $3}')

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
      slack "📌 *already fixed on main*: \`$first_test\` ($workflow)
fixed by $upstream · waiting for the next run that contains \`${fsha:0:10}\` to confirm it is green"
    else
      log "  no fix: $reason"
      $FIXLIB mark --state no_fix --attempt --extra "$(jq -nc --arg p "$pdir" --arg r "$reason" '{proposal:$p, reason:$r}')" "${sigs[@]}"
      slack "🛠 autofix looked at \`$first_test\` ($workflow): *no safe fix*. $reason"
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
    slack "🛠 autofix drafted a fix for \`$first_test\` but the safety guard rejected it: $(jq -r '.reasons | join("; ")' <<<"$guard"). Proposal: \`$pdir\`"
    git -C "$FIX_WORKTREE" reset -q --hard && git -C "$FIX_WORKTREE" clean -fdq
    continue
  fi

  jobs_list=$(jq -c '[to_entries[].value.sku_jobs[]] | unique' <<<"$recs")
  $FIXLIB dispatch --workflow "$workflow" --jobs "$jobs_list" --repo "$FIX_WORKTREE" > "$pdir/dispatch.json"
  thr=""; [[ "$(jq -r .threshold_change "$pdir/verdict.json")" == "true" ]] && thr=" [threshold]"
  title="$PR_TITLE_PREFIX$thr $v_title"
  printf '%s\n' "$title" > "$pdir/pr_title.txt"
  render() {
    jq -n --slurpfile v "$pdir/verdict.json" --argjson recs "$recs" --slurpfile d "$pdir/dispatch.json" \
          --argjson rel "$related" --arg repo "$REPO" --arg model "$FIX_MODEL" --arg ci "$CI_STATUS_CONTEXT" \
          '{verdict: $v[0], records: [$recs | to_entries[].value], sigs: ($recs | keys), dispatch: $d[0],
            related_prs: $rel, repo: $repo, model: $model, ci_ctx: $ci}' > "$pdir/meta.json"
    $FIXLIB render --meta "$pdir/meta.json" > "$pdir/pr_body.md"
  }
  render
  files=$(jq -r '.files | join(", ")' <<<"$guard")
  conf=$(jq -r .confidence "$pdir/verdict.json"); appr=$(jq -r .approach "$pdir/verdict.json")

  if [[ "$FIX_MODE" != "live" ]]; then
    $FIXLIB mark --state proposed_dryrun --attempt --count-daily \
      --extra "$(jq -nc --arg p "$pdir" --arg t "$title" '{proposal:$p, verdict_title:$t}')" "${sigs[@]}"
    log "  DRY RUN — proposal written to $pdir"
    slack "🛠 *autofix (dry run)* would open a draft PR: *$title*
• failing: \`$first_test\` ($workflow, streak $(jq -r '[.[].streak] | max' <<<"$recs"))
• $appr, confidence $conf, $(jq -r '.added + .deleted' <<<"$guard") lines in $files
• proposal: \`$pdir\`"
    git -C "$FIX_WORKTREE" reset -q --hard && git -C "$FIX_WORKTREE" clean -fdq
    continue
  fi

  # ---------------- live: branch, commit, push, draft PR, dispatch ----------------
  branch="$BRANCH_PREFIX/$sig8-$(slugify "$v_title")"
  git -C "$FIX_WORKTREE" checkout -q -b "$branch"
  git -C "$FIX_WORKTREE" commit -q --no-verify -F - <<EOF
$v_title

$(jq -r .root_cause "$pdir/verdict.json")

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

  # Dispatch and resolve run ids (gh workflow run does not print one).
  disp_out="[]"
  while IFS= read -r p; do
    wf=$(jq -r .workflow <<<"$p")
    if [[ "$(jq -r '.inputs == null' <<<"$p")" == "true" ]]; then disp_out=$(jq -c --argjson p "$p" '. + [$p]' <<<"$disp_out"); continue; fi
    mapfile -t fargs < <(jq -r '.inputs | to_entries[] | "-f", "\(.key)=\(.value)"' <<<"$p")
    t0=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    gh workflow run "$wf" -R "$REPO" --ref "$branch" "${fargs[@]}"
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
  render
  gh pr edit "$pr_url" --body-file "$pdir/pr_body.md" >/dev/null

  $FIXLIB mark --state pr_open --attempt --count-daily \
    --extra "$(jq -nc --arg u "$pr_url" --arg n "$pr_num" --arg b "$branch" --arg p "$pdir" --argjson d "$disp_out" \
               '{pr:{url:$u, number:($n|tonumber), branch:$b}, proposal:$p, dispatched:$d}')" "${sigs[@]}"
  log "  opened draft $pr_url"
  slack "🛠️ *opened* draft PR #$pr_num (needs human review): *$title*
• failing: \`$first_test\` ($workflow)
• $appr, confidence $conf · targeted CI dispatched
$pr_url"
  git -C "$FIX_WORKTREE" checkout -q -f --detach origin/main
done < <(jq -c '.groups[]' <<<"$elig")

log "==== fixer tick done ===="
