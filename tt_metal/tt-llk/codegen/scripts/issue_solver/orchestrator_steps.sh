#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# orchestrator_steps.sh — the executable steps of the issue-solver orchestrators.
#
# Every pipeline step the issue-solver orchestrators run
# (agents/issue-solver/orchestrator.md single-arch, orchestrator-multi.md
# multi-arch) is a function here. The playbooks source this file and call one
# function per step, passing per-run values as arguments — so no bash is
# hand-assembled in the prompt and no state is trusted to survive between Bash
# tool calls (exported env vars only live for one Bash invocation).
#
# RUN_MODE ("single" | "multi") in the run-state file selects the divergent
# behaviour (arch profile, arch_results aggregation, finalize patch shape). The
# ~80% of logic that is identical between the two orchestrators is shared.
#
# Usage: run with cwd = $WORKTREE_DIR/tt_metal/tt-llk, then
#     source codegen/scripts/issue_solver/orchestrator_steps.sh
#     execute_step_setup_run
#
# Mirrors codegen/scripts/quasar/orchestrator_steps.sh (same helper conventions).

# Physical scripts dir (…/codegen/scripts), resolved from this file's location so
# python helpers are found regardless of cwd. `codegen` is a symlink in the
# worktree; following it to the source copy is fine — same code.
_ORCH_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- out-of-space (ENOSPC) guard -------------------------------------------
# True when a captured error string carries the out-of-space signature.
_is_enospc() { printf '%s' "$1" | grep -qiE 'no space left on device|errno 28|enospc'; }
# High-priority abort banner for the orchestrator, on stderr so it stands out.
_no_space_banner() {
    printf '%s\n' \
      "################################################################" \
      "## NO SPACE LEFT ON DEVICE — HIGH PRIORITY, STOP NOW" \
      "## Spawn no agents and run no further steps. Report this run" \
      "## FAILED with reason: no space on device. Then run exactly:" \
      "##   execute_step_report_no_space \"<current step>\"" \
      "################################################################" >&2
}
# Run a disk-writing command; on an out-of-space failure print the abort banner
# and return 28, otherwise pass its output and exit code through unchanged.
_disk_guard() {
    local out rc
    out="$("$@" 2>&1)"; rc=$?
    [ -n "$out" ] && printf '%s\n' "$out"
    if [ "$rc" -ne 0 ] && _is_enospc "$out"; then _no_space_banner; return 28; fi
    return "$rc"
}

# --- env-free state/run-json helpers ---------------------------------------
# Worktree root from cwd (cwd == <wt>/tt_metal/tt-llk). Subshell: no cwd change.
_wt()  { ( cd ../.. && pwd ); }
# LOG_DIR is the one bootstrap key kept in the worktree file.
_LOG() { python "$_ORCH_SCRIPTS/state.py" --worktree-dir "$(_wt)" get LOG_DIR; }
# Run-state accessors — `_L` is set once at the top of each function below.
# ss/rj write to disk, so they run through _disk_guard; sg only reads.
sg()   { python "$_ORCH_SCRIPTS/state.py" --log-dir "$_L" get "$@"; }
ss()   { _disk_guard python "$_ORCH_SCRIPTS/state.py" --log-dir "$_L" set "$@"; }
rj()   { local sub="$1"; shift; _disk_guard python "$_ORCH_SCRIPTS/run_json_writer.py" "$sub" --log-dir "$_L" "$@"; }
# refresh_cost.sh recovers everything itself; hand it LOG_DIR + the session id.
refresh_cost() {
    local sid pcwd
    sid="$(sg SESSION_ID 2>/dev/null || true)"; pcwd="$(sg PROJECT_CWD 2>/dev/null || true)"
    python "$_ORCH_SCRIPTS/session_cost.py" --since "$(sg START_TIME)" --log-dir "$_L" \
        ${sid:+--session-id "$sid" --project-cwd "$pcwd"} >/dev/null 2>&1 || true
}

# Pipeline stages shown on the dashboard. The two orchestrators differ only in a
# few `desc` strings ("all target arches" vs "scope"); keep both verbatim so the
# dashboard reads identically to the pre-refactor runs.
_PIPELINE_STEPS_MULTI='[
  {"id":"analyzer","name":"Analyze","desc":"Understand the issue and all target arches"},
  {"id":"arch_lookup","name":"Research","desc":"Look up architecture facts only when needed"},
  {"id":"writer","name":"Fix","desc":"Plan and implement one coordinated multi-arch fix"},
  {"id":"tester","name":"Test","desc":"Run the tt-llk Layer-1 suite for each target arch"},
  {"id":"metal_test","name":"Metal Test","desc":"Build+run the unit_tests_llk gtest suite for Layer-2/3/4 changes (same backend)"},
  {"id":"review","name":"Review","desc":"Senior LLK review of the shared fix diff (loop, no PR)"},
  {"id":"perf","name":"Perf","desc":"Measure cycle counts vs baseline per BH/WH arch (local only)"},
  {"id":"fix_tests","name":"Retry","desc":"Debug and update the shared fix after a test, review, or perf failure"}
]'
_PIPELINE_STEPS_SINGLE='[
  {"id":"analyzer","name":"Analyze","desc":"Understand the issue and scope"},
  {"id":"arch_lookup","name":"Research","desc":"Look up architecture facts only when needed"},
  {"id":"writer","name":"Fix","desc":"Plan and implement the smallest fix"},
  {"id":"tester","name":"Test","desc":"Run the tt-llk Layer-1 suite"},
  {"id":"metal_test","name":"Metal Test","desc":"Build+run the unit_tests_llk gtest for Layer-2/3/4 changes (same backend)"},
  {"id":"review","name":"Review","desc":"Senior LLK review of the fix diff (loop, no PR)"},
  {"id":"perf","name":"Perf","desc":"Measure cycle counts vs baseline (BH/WH local only)"},
  {"id":"fix_tests","name":"Retry","desc":"Debug and update the fix after a test, review, or perf failure"}
]'
# RUN_KIND=review: an address-comments round on an open PR. No analyze stage —
# scope and verification route are inherited from the solve that produced the PR.
_PIPELINE_STEPS_REVIEW='[
  {"id":"addresser","name":"Address","desc":"Turn the PR review feedback into code changes"},
  {"id":"tester","name":"Test","desc":"Run the tt-llk Layer-1 suite"},
  {"id":"metal_test","name":"Metal Test","desc":"Build+run the unit_tests_llk gtest for Layer-2/3/4 changes"},
  {"id":"review","name":"Review","desc":"Senior LLK review of the updated diff (loop, no PR)"},
  {"id":"perf","name":"Perf","desc":"Measure cycle counts only when a disposition asks for it"},
  {"id":"fix_tests","name":"Retry","desc":"Debug and update the review fix after a test or review failure"}
]'

# ===========================================================================
# Any step — emit a mid-step progress message (does not change the step).
# Arg: <message>.
# ===========================================================================
execute_step_message() {
    local _L; _L="$(_LOG)"
    rj message --message "$1"
}

# ===========================================================================
# Any step — out-of-space terminal handler. Call this the moment a step prints
# the NO SPACE banner. Retries the run.json failed-finalize every 30s for up to
# 10 minutes until the write lands, appends the runs.jsonl entry, then returns.
# Arg: <step where space ran out>.
# ===========================================================================
execute_step_report_no_space() {
    local _L; _L="$(_LOG)"
    local where="${1:-unknown}" lb deadline=$(( SECONDS + 600 )) attempt=0 rc
    lb="$(sg LOGS_BASE)"
    while :; do
        attempt=$(( attempt + 1 ))
        python "$_ORCH_SCRIPTS/run_json_writer.py" finalize \
            --log-dir "$_L" \
            --status failed \
            --final-result test_failure \
            --solver-state not_working \
            --final-message "Run aborted at ${where} — no space left on device" \
            --patch-json '{"obstacle":"no space left on device"}' >/dev/null 2>&1
        rc=$?
        if [ "$rc" -eq 0 ]; then
            python "$_ORCH_SCRIPTS/issue_solver_run_utils.py" upsert-runs-jsonl \
                --log-dir "$_L" --runs-jsonl "${lb}/runs.jsonl" >/dev/null 2>&1 || true
            echo "NO_SPACE_REPORTED: run.json finalized failed (no space on device) after ${attempt} attempt(s)"
            return 0
        fi
        if [ "$SECONDS" -ge "$deadline" ]; then
            _no_space_banner
            echo "NO_SPACE_UNREPORTABLE: run.json still unwritable after 10 min / ${attempt} attempts — device full" >&2
            return 28
        fi
        echo "NO_SPACE_RETRY: finalize failed (attempt ${attempt}) — device full, retry in 30s" >&2
        sleep 30
    done
}

# ===========================================================================
# Input — validate the router's handoff. Arg: absolute worktree dir.
# Reads the bootstrap keys the router wrote to the worktree state file.
# ===========================================================================
execute_step_validate_input() {
    local wt="$1"
    [ -n "$wt" ] && [ -d "$wt" ] || { echo "REJECT: WORKTREE_DIR missing or not a directory: '$wt'"; return 1; }
    cd "$wt/tt_metal/tt-llk" || { echo "REJECT: cannot cd into $wt/tt_metal/tt-llk"; return 1; }

    local S="$_ORCH_SCRIPTS" mode num title wb tb clb cpr arches arch dirty ok=1
    mode="$(python "$S/state.py" --worktree-dir "$wt" get RUN_MODE)"
    num="$(python "$S/state.py" --worktree-dir "$wt" get ISSUE_NUMBER)"
    title="$(python "$S/state.py" --worktree-dir "$wt" get ISSUE_TITLE)"
    wb="$(python "$S/state.py" --worktree-dir "$wt" get WORKTREE_BRANCH)"
    tb="$(python "$S/state.py" --worktree-dir "$wt" get TEST_BACKEND)"
    clb="$(python "$S/state.py" --worktree-dir "$wt" get CREATE_LOCAL_BRANCH)"
    cpr="$(python "$S/state.py" --worktree-dir "$wt" get CREATE_PR)"
    arch="$(python "$S/state.py" --worktree-dir "$wt" get TARGET_ARCH)"
    arches="$(python "$S/state.py" --worktree-dir "$wt" get TARGET_ARCHES)"

    { [ "$mode" = "single" ] || [ "$mode" = "multi" ]; } || { echo "REJECT: RUN_MODE must be single|multi (got '$mode')"; ok=0; }
    printf '%s' "$num" | grep -qE '^[0-9]+$' || { echo "REJECT: ISSUE_NUMBER must be numeric (got '$num')"; ok=0; }
    [ -n "$title" ] || { echo "REJECT: ISSUE_TITLE is empty"; ok=0; }
    [ -n "$wb" ] || { echo "REJECT: WORKTREE_BRANCH is empty"; ok=0; }
    { [ "$tb" = "local" ] || [ "$tb" = "ttsim" ]; } || { echo "REJECT: TEST_BACKEND must be local|ttsim (got '$tb')"; ok=0; }
    { [ "$clb" = "yes" ] || [ "$clb" = "no" ]; } || { echo "REJECT: CREATE_LOCAL_BRANCH must be yes|no (got '$clb')"; ok=0; }
    { [ "$cpr" = "yes" ] || [ "$cpr" = "no" ]; } || { echo "REJECT: CREATE_PR must be yes|no (got '$cpr')"; ok=0; }
    if [ "$mode" = "single" ]; then
        [ -n "$arch" ] || { echo "REJECT: TARGET_ARCH is empty (single-arch run)"; ok=0; }
    else
        [ -n "$arches" ] || { echo "REJECT: TARGET_ARCHES is empty (multi-arch run)"; ok=0; }
    fi
    if ! dirty="$(git -C "$wt" status --porcelain --untracked-files=all 2>/dev/null)"; then
        echo "REJECT: cannot inspect worktree status"
        ok=0
    elif [ -n "$dirty" ]; then
        echo "REJECT: issue-solver worktree must start clean; unexpected paths:"
        printf '%s\n' "$dirty"
        ok=0
    fi
    [ "$ok" = 1 ] || return 1
    echo "OK: RUN_MODE=$mode ISSUE=#$num arch=${arch:-$arches} TEST_BACKEND=$tb"
}

# ===========================================================================
# Step 0 — validate environment prerequisites (settings.validate()).
# ===========================================================================
execute_step_validate_env() {
    ( cd "$_ORCH_SCRIPTS/../.." \
        && PYTHONPATH=.. python -c "from codegen.config.settings import settings; issues = settings.validate(); [print(f'ISSUE: {i}') for i in issues]; exit(1) if issues else print('Environment OK')" 2>/dev/null ) \
        || echo "validate_env: settings.validate() unavailable or reported issues (non-fatal for issue-solver)"
}

# ===========================================================================
# Router step (RUN_KIND=review) — seed the bootstrap state for an address-comments
# round from the solve run that produced the PR. That run's state.json already
# holds the issue text, scope and every verification key, so copying them is what
# lets the round reuse the route_verification → tester tail unchanged.
#
# Arg: <worktree_dir>. Environment (set by the dashboard's dispatcher):
#   CODEGEN_SOURCE_RUN_DIR   the solve run's LOG_DIR                    (required)
#   CODEGEN_REVIEW_INPUT     the reviewer-feedback JSON document        (required)
#   CODEGEN_PR_NUMBER        the PR being updated                       (required)
#   CODEGEN_REVIEW_ARCHES    JSON array; overrides the source arches    (optional)
#   CODEGEN_REVIEW_TEST_BACKEND  local|ttsim                            (default local)
#   CODEGEN_PR_HEAD_SHA      the commit this round is based on          (optional)
# ===========================================================================
execute_step_seed_review_state() {
    local wt="$1" S="$_ORCH_SCRIPTS" src="${CODEGEN_SOURCE_RUN_DIR:-}"
    [ -n "$wt" ] && [ -d "$wt" ] || { echo "REJECT: WORKTREE_DIR is missing: '$wt'"; return 1; }
    [ -n "$src" ] && [ -f "$src/state.json" ] || {
        echo "REJECT: CODEGEN_SOURCE_RUN_DIR must be a solve run dir with state.json (got '${src:-unset}')"
        return 1; }
    [ -n "${CODEGEN_REVIEW_INPUT:-}" ] && [ -f "${CODEGEN_REVIEW_INPUT}" ] || {
        echo "REJECT: CODEGEN_REVIEW_INPUT must be the reviewer-feedback JSON (got '${CODEGEN_REVIEW_INPUT:-unset}')"
        return 1; }

    # One python pass builds the whole patch, one `set-many` writes it. Copying
    # these keys is what lets the round reuse the route_verification → tester tail
    # without re-analyzing the issue.
    python - "$src" <<'PY' | _disk_guard python "$S/state.py" --worktree-dir "$wt" set-many
import json, os, sys
state = json.load(open(os.path.join(sys.argv[1], "state.json")))
pr = os.environ.get("CODEGEN_PR_NUMBER", "")
num = str(state.get("ISSUE_NUMBER") or "")
backend = os.environ.get("CODEGEN_REVIEW_TEST_BACKEND") or "local"
if not pr.isdigit():          raise SystemExit(f"REJECT: CODEGEN_PR_NUMBER not numeric: {pr!r}")
if not num.isdigit():         raise SystemExit("REJECT: source run has no numeric ISSUE_NUMBER")
if backend not in ("local", "ttsim"): raise SystemExit("REJECT: TEST_BACKEND must be local|ttsim")

raw = (os.environ.get("CODEGEN_REVIEW_ARCHES") or "").strip()
arches = json.loads(raw) if raw else (state.get("TARGET_ARCHES_JSON")
         or ([state["TARGET_ARCH"]] if state.get("TARGET_ARCH") else []))
if isinstance(arches, str):
    arches = json.loads(arches)
aliases = {"bh": "blackhole", "wh": "wormhole", "qsr": "quasar"}
seen, targets = set(), []
for value in arches:
    arch = aliases.get(str(value).strip().lower(), str(value).strip().lower())
    if arch not in {"blackhole", "wormhole", "quasar"}:
        raise SystemExit(f"REJECT: unknown target arch: {value}")
    if arch not in seen:
        seen.add(arch); targets.append(arch)
if not targets:
    raise SystemExit("REJECT: source run recorded no target arch")

patch = {k: state.get(k) or "" for k in
         ("ISSUE_TITLE", "ISSUE_BODY", "ISSUE_LABELS", "ISSUE_COMMENTS", "ISSUE_URL")}
patch.update({
    "RUN_KIND": "review", "ISSUE_NUMBER": num, "TEST_BACKEND": backend,
    # The round updates an existing PR: it commits locally, never opens one.
    "CREATE_LOCAL_BRANCH": "yes", "CREATE_PR": "no",
    "PR_NUMBER": pr, "PR_HEAD_SHA": os.environ.get("CODEGEN_PR_HEAD_SHA", ""),
    "REVIEW_INPUT": os.environ["CODEGEN_REVIEW_INPUT"],
    "SOURCE_RUN_DIR": sys.argv[1], "SOURCE_RUN_ID": state.get("RUN_ID") or "",
})
# The same "1 → single, N → multi" rule execute_step_setup_run applies.
single = len(targets) == 1
patch["RUN_MODE"] = "single" if single else "multi"
patch["TARGET_ARCH" if single else "TARGET_ARCHES"] = targets[0] if single else json.dumps(targets)
key = "TTSIM_SO_PATH" if single else "TTSIM_SO_PATHS"
if state.get(key):
    patch[key] = state[key]
json.dump(patch, sys.stdout)
print(f"OK: PR #{pr}, issue #{num}, arches={targets}, backend={backend}", file=sys.stderr)
PY
    [ "${PIPESTATUS[0]}" -eq 0 ] || return 1
}

# ===========================================================================
# Step 0 (RUN_KIND=review) — carry the review identity into the run state and
# import the source solve's artifacts. Run right after execute_step_setup_run.
# The analysis artifact is what route_verification parses, so importing it gives
# the round a route without re-analyzing; the addresser may amend it when the
# review asks for coverage the solve did not add.
# ===========================================================================
execute_step_setup_review_run() {
    local _L; _L="$(_LOG)"
    local S="$_ORCH_SCRIPTS" wt src num; wt="$(_wt)"
    src="$(python "$S/state.py" --worktree-dir "$wt" get SOURCE_RUN_DIR)"
    num="$(sg ISSUE_NUMBER)"
    [ -n "$src" ] && [ -d "$src" ] || {
        echo "REJECT: SOURCE_RUN_DIR is missing from bootstrap state; run execute_step_seed_review_state first" >&2
        return 1; }

    ss PR_NUMBER      "$(python "$S/state.py" --worktree-dir "$wt" get PR_NUMBER)"
    ss PR_HEAD_SHA    "$(python "$S/state.py" --worktree-dir "$wt" get PR_HEAD_SHA)"
    ss REVIEW_INPUT   "$(python "$S/state.py" --worktree-dir "$wt" get REVIEW_INPUT)"
    ss SOURCE_RUN_DIR "$src"
    ss SOURCE_RUN_ID  "$(python "$S/state.py" --worktree-dir "$wt" get SOURCE_RUN_ID)"
    # Perf off unless a disposition asks: a baseline measurement doubles wall clock.
    ss PERF_REQUESTED 0 --json

    local imported=0 f analysis fix_plan
    _disk_guard mkdir -p codegen/artifacts || return $?
    analysis="$src/issue_${num}_analysis.md"
    fix_plan="$src/issue_${num}_fix_plan.md"
    if [ ! -f "$analysis" ]; then
        ss OBSTACLE "review round could not import the source run's analysis artifact"
        execute_step_mark_status failed
        echo "REJECT: missing issue_${num}_analysis.md in $src — the round has no verification route" >&2
        return 1
    fi
    for f in "$analysis" "$fix_plan"; do
        [ -f "$f" ] || continue
        cp "$f" codegen/artifacts/ 2>/dev/null && imported=$((imported + 1))
        cp "$f" "$_L/" 2>/dev/null || true
    done
    # The reviewer agent reads the solve's own review verdict for continuity.
    cp "$src/review_result.json" "$_L/source_review_result.json" 2>/dev/null || true

    rj message --message "Addressing review on PR #$(sg PR_NUMBER); imported ${imported} artifact(s) from $(sg SOURCE_RUN_ID)"
    echo "PR=#$(sg PR_NUMBER) SOURCE_RUN_ID=$(sg SOURCE_RUN_ID) ARTIFACTS=${imported}"
}

# ===========================================================================
# Step 1 (RUN_KIND=review) — advance to the addresser. Optional arg: prev agent
# (default "addresser" on the first pass; pass "fix_tests" on a debug retry).
# ===========================================================================
execute_step_advance_addresser() {
    local _L; _L="$(_LOG)"
    local agent="${1:-addresser}" pr n
    pr="$(sg PR_NUMBER)"
    n="$(python -c "import json,sys;print(len(json.load(open(sys.argv[1]))['actionable_threads']))" "$(sg REVIEW_INPUT)" 2>/dev/null || echo "?")"
    rj advance --new-step "addresser" \
        --new-message "Addressing ${n} review thread(s) on PR #${pr}" \
        --prev-result "success" --prev-message "Review feedback collected" --agent "$agent"
    ss PREVIOUS_AGENT "addresser"
}

# ===========================================================================
# Step 1 (RUN_KIND=review) — validate and record the addresser's dispositions.
# Every actionable thread needs exactly one; a missing one used to degrade into a
# generic reply the agent had never considered, so this fails the run instead.
# Also enforces the reply contract: length, and no commit sha (the dashboard owns
# attribution, and a model-written sha is routinely the wrong one).
# ===========================================================================
execute_step_record_review_dispositions() {
    local _L; _L="$(_LOG)"
    local max_chars="${1:-600}" out
    out="$(LOG_DIR="$_L" REVIEW_INPUT="$(sg REVIEW_INPUT)" MAX_CHARS="$max_chars" python - <<'PY'
import json, os, re, sys

log_dir = os.environ["LOG_DIR"]
path = os.path.join(log_dir, "review_dispositions.json")
try:
    doc = json.load(open(path))
except FileNotFoundError:
    raise SystemExit(f"MISSING: the addresser wrote no {path}")
except (ValueError, OSError) as e:
    raise SystemExit(f"INVALID: {path} is not readable JSON ({e})")

entries = doc.get("threads") if isinstance(doc, dict) else None
if not isinstance(entries, list):
    raise SystemExit(f"INVALID: {path} must be an object with a 'threads' array")

required = {
    str(t["comment_id"])
    for t in json.load(open(os.environ["REVIEW_INPUT"]))["actionable_threads"]
}
max_chars = int(os.environ["MAX_CHARS"])
allowed = {"changed", "no_change", "disagree", "deferred"}
by_id, problems = {}, []
for entry in entries:
    if not isinstance(entry, dict):
        problems.append("a thread entry is not an object"); continue
    cid = str(entry.get("comment_id") or "")
    action = str(entry.get("action") or "").strip()
    reply = str(entry.get("reply") or "").strip()
    if cid not in required:
        continue                      # extra ids (summaries, stale threads) are ignored
    if cid in by_id:
        problems.append(f"{cid}: duplicate disposition")
        continue
    if action not in allowed:
        problems.append(f"{cid}: action must be one of {sorted(allowed)} (got {action!r})")
    if not reply:
        problems.append(f"{cid}: reply is empty")
    elif len(reply) > max_chars:
        problems.append(f"{cid}: reply is {len(reply)} chars (limit {max_chars})")
    elif re.search(r"\b[0-9a-f]{7,40}\b", reply):
        problems.append(f"{cid}: reply cites a commit sha; the dashboard adds that")
    by_id[cid] = entry

missing = sorted(required - set(by_id))
if missing:
    problems.append("no disposition for actionable thread(s): " + ", ".join(missing))
if problems:
    raise SystemExit("INVALID_DISPOSITIONS:\n  " + "\n  ".join(problems))

perf = any(e.get("perf_relevant") for e in by_id.values())
normalized = {"version": 1, "threads": [by_id[k] for k in sorted(by_id)]}
with open(path, "w", encoding="utf-8") as f:
    json.dump(normalized, f, indent=2)
print(json.dumps({
    "count": len(by_id),
    "changed": sum(1 for e in by_id.values() if e.get("action") == "changed"),
    "perf_relevant": perf,
}))
PY
)" || { echo "$out" >&2; ss OBSTACLE "invalid review dispositions"; return 1; }

    ss PERF_REQUESTED "$(printf '%s' "$out" | python -c "import json,sys;print(int(json.load(sys.stdin)['perf_relevant']))")" --json
    rj metric --patch-json "{\"review_dispositions\": $out}"
    echo "$out"
}

# ===========================================================================
# Step 0 — compute run identity/timing/dirs, resolve the log + knowledge roots,
# seed counters, normalize the arch list, snapshot playbooks, and capture the
# session. Reads the router handoff from the worktree state file + the ambient
# CODEGEN_* env passthroughs. Writes everything to the run-state file and sets
# the LOG_DIR bootstrap key in the worktree file.
# ===========================================================================
execute_step_setup_run() {
    local S="$_ORCH_SCRIPTS" wt; wt="$(_wt)"

    # --- router handoff (worktree bootstrap file) ---------------------------
    local MODE ISSUE_NUMBER ISSUE_TITLE ISSUE_BODY ISSUE_LABELS ISSUE_COMMENTS ISSUE_URL
    local WORKTREE_BRANCH TEST_BACKEND CREATE_LOCAL_BRANCH CREATE_PR RUN_KIND
    MODE="$(python "$S/state.py" --worktree-dir "$wt" get RUN_MODE)"; MODE="${MODE:-single}"
    # "issue" (solve, default) | "review". Selects the pipeline shape + first step.
    RUN_KIND="$(python "$S/state.py" --worktree-dir "$wt" get RUN_KIND)"; RUN_KIND="${RUN_KIND:-issue}"
    ISSUE_NUMBER="$(python "$S/state.py" --worktree-dir "$wt" get ISSUE_NUMBER)"
    ISSUE_TITLE="$(python "$S/state.py" --worktree-dir "$wt" get ISSUE_TITLE)"
    WORKTREE_BRANCH="$(python "$S/state.py" --worktree-dir "$wt" get WORKTREE_BRANCH)"
    TEST_BACKEND="$(python "$S/state.py" --worktree-dir "$wt" get TEST_BACKEND)"
    CREATE_LOCAL_BRANCH="$(python "$S/state.py" --worktree-dir "$wt" get CREATE_LOCAL_BRANCH)"
    CREATE_PR="$(python "$S/state.py" --worktree-dir "$wt" get CREATE_PR)"

    # --- arch profile + canonical dashboard project id -----------------------
    # Architecture is run metadata. Single- and multi-arch runs share the same
    # archive, matching Quasar's one-project/one-root dashboard layout.
    local DASHBOARD_PROJECT_ID TARGET_ARCH TARGET_ARCHES_JSON ARCH_COUNT ARCH_PROFILES_JSON
    DASHBOARD_PROJECT_ID="issue_solver"
    if [ "$MODE" = "single" ]; then
        TARGET_ARCH="$(python "$S/state.py" --worktree-dir "$wt" get TARGET_ARCH)"
        TARGET_ARCHES_JSON="$(python -c "import json,sys; print(json.dumps([sys.argv[1]]))" "$TARGET_ARCH")"
    else
        TARGET_ARCHES_JSON="$(python - "$(python "$S/state.py" --worktree-dir "$wt" get TARGET_ARCHES)" <<'PY'
import json, sys
raw = sys.argv[1]
values = json.loads(raw) if raw.strip().startswith("[") else [p.strip() for p in raw.split(",") if p.strip()]
aliases = {"bh": "blackhole", "wh": "wormhole", "qsr": "quasar"}
seen, arches = set(), []
for value in values:
    arch = aliases.get(str(value).strip().lower(), str(value).strip().lower())
    if arch not in {"blackhole", "wormhole", "quasar"}:
        raise SystemExit(f"unknown target arch: {value}")
    if arch not in seen:
        seen.add(arch); arches.append(arch)
print(json.dumps(arches))
PY
)"
    fi
    ARCH_COUNT="$(python -c "import json,sys; print(len(json.loads(sys.argv[1])))" "$TARGET_ARCHES_JSON")"
    # Per-arch profile map (LLK dir / reference arch / reference LLK dir).
    ARCH_PROFILES_JSON="$(python - "$TARGET_ARCHES_JSON" <<'PY'
import json, sys
profile = {
    "blackhole": {"llk_dir": "tt_llk_blackhole",    "ref_arch": "wormhole", "ref_llk_dir": "tt_llk_wormhole_b0"},
    "wormhole":  {"llk_dir": "tt_llk_wormhole_b0",  "ref_arch": "",         "ref_llk_dir": ""},
    "quasar":    {"llk_dir": "tt_llk_quasar",       "ref_arch": "blackhole","ref_llk_dir": "tt_llk_blackhole"},
}
arches = json.loads(sys.argv[1])
out = {}
for a in arches:
    if a not in profile:
        raise SystemExit(f"unknown target arch: {a}")
    out[a] = profile[a]
print(json.dumps(out))
PY
)"

    # --- log + knowledge roots (resolved against the MAIN checkout) ---------
    # Capture the exported override before declaring the function-local value;
    # `local CODEGEN_LOGS_ROOT` by itself would shadow and discard it.
    local configured_logs_root="${CODEGEN_LOGS_ROOT:-}"
    local CODEGEN_LOGS_ROOT="$configured_logs_root" LOGS_BASE PR_REVIEW_KNOWLEDGE_DIR MAIN_REPO_ROOT
    if [ -z "$CODEGEN_LOGS_ROOT" ]; then
        if [ -d /proj_sw/user_dev/llk_code_gen ]; then
            CODEGEN_LOGS_ROOT="/proj_sw/user_dev/llk_code_gen"
        else
            MAIN_REPO_ROOT="$(dirname "$(git -C "$wt" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)" 2>/dev/null)"
            if [ -n "$MAIN_REPO_ROOT" ] && [ -d "$MAIN_REPO_ROOT/tt_metal/tt-llk/codegen" ]; then
                CODEGEN_LOGS_ROOT="${MAIN_REPO_ROOT}/tt_metal/tt-llk/codegen/logs"
            else
                CODEGEN_LOGS_ROOT="$wt/tt_metal/tt-llk/codegen/logs"
            fi
        fi
    fi
    LOGS_BASE="${CODEGEN_LOGS_ROOT}/${DASHBOARD_PROJECT_ID}"

    if [ -n "${CODEGEN_PR_REVIEW_KNOWLEDGE:-}" ] && [ -d "${CODEGEN_PR_REVIEW_KNOWLEDGE}" ]; then
        PR_REVIEW_KNOWLEDGE_DIR="${CODEGEN_PR_REVIEW_KNOWLEDGE}"
    elif [ -d "${CODEGEN_LOGS_ROOT}/dashboard/pr_review/knowledge" ]; then
        PR_REVIEW_KNOWLEDGE_DIR="${CODEGEN_LOGS_ROOT}/dashboard/pr_review/knowledge"
    elif [ -d /proj_sw/user_dev/llk_code_gen/dashboard/pr_review/knowledge ]; then
        PR_REVIEW_KNOWLEDGE_DIR="/proj_sw/user_dev/llk_code_gen/dashboard/pr_review/knowledge"
    else
        MAIN_REPO_ROOT="${MAIN_REPO_ROOT:-$(dirname "$(git -C "$wt" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)" 2>/dev/null)}"
        if [ -n "$MAIN_REPO_ROOT" ] && [ -d "$(dirname "$MAIN_REPO_ROOT")/llk_code_gen/dashboard/pr_review/knowledge" ]; then
            PR_REVIEW_KNOWLEDGE_DIR="$(dirname "$MAIN_REPO_ROOT")/llk_code_gen/dashboard/pr_review/knowledge"
        else
            PR_REVIEW_KNOWLEDGE_DIR=""
        fi
    fi

    # --- identity + timing --------------------------------------------------
    local START_TIME RUN_ID LOG_DIR GIT_COMMIT GIT_BRANCH CODEGEN_VERSION suffix
    START_TIME="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    [ "$MODE" = "multi" ] && suffix="_multi" || suffix=""
    # Tag the id so review runs stay distinguishable from the solve's in the dirs.
    [ "$RUN_KIND" = "review" ] && suffix="${suffix}_review"
    RUN_ID="$(date +%Y-%m-%d)_issue_${ISSUE_NUMBER}${suffix}_$(head -c 4 /dev/urandom | xxd -p)"
    LOG_DIR="${LOGS_BASE}/${RUN_ID}"
    GIT_COMMIT="$(git -C "$wt" rev-parse HEAD 2>/dev/null || echo unknown)"
    GIT_BRANCH="$(git -C "$wt" branch --show-current 2>/dev/null || echo "$WORKTREE_BRANCH")"
    CODEGEN_VERSION="$(tr -d '[:space:]' < codegen/agents/issue-solver/VERSION 2>/dev/null || echo "")"

    # --- PERF_GOAL keyword guess (refined later from analyzer perf_intent) --
    local PERF_GOAL ISSUE_BODY ISSUE_LABELS ISSUE_COMMENTS
    ISSUE_BODY="$(python "$S/state.py" --worktree-dir "$wt" get ISSUE_BODY)"
    ISSUE_LABELS="$(python "$S/state.py" --worktree-dir "$wt" get ISSUE_LABELS)"
    ISSUE_COMMENTS="$(python "$S/state.py" --worktree-dir "$wt" get ISSUE_COMMENTS)"
    if printf '%s' "${ISSUE_TITLE} ${ISSUE_LABELS} ${ISSUE_BODY}" | grep -qiE \
        'perf|performance|optimi|speed|slow|cycles|latency|throughput|regression|recover'; then
        PERF_GOAL=improve
    else
        PERF_GOAL=no_regress
    fi

    _disk_guard mkdir -p "$LOG_DIR/instructions" codegen/artifacts || return $?
    # Snapshot the playbooks this run executed. `review/` is nested, so a flat glob
    # would skip it and the round would archive instructions it never ran.
    cp codegen/agents/issue-solver/*.md "$LOG_DIR/instructions/" 2>/dev/null || true
    if [ "$RUN_KIND" = "review" ]; then
        for f in codegen/agents/issue-solver/review/*.md; do
            [ -f "$f" ] && cp "$f" "$LOG_DIR/instructions/review-$(basename "$f")" 2>/dev/null || true
        done
    fi
    cp .claude/CLAUDE.md "$LOG_DIR/instructions/tt-llk-CLAUDE.md" 2>/dev/null || true
    cp -R .claude/skills "$LOG_DIR/instructions/claude-skills" 2>/dev/null || true

    # LOG_DIR and RUN_ID are bootstrap identity — write them to the worktree file
    # so later steps and queue dispatch recover them with no persistent shell env.
    _disk_guard python "$S/state.py" --worktree-dir "$wt" set LOG_DIR "$LOG_DIR" || return $?
    _disk_guard python "$S/state.py" --worktree-dir "$wt" set RUN_ID "$RUN_ID" || return $?

    # --- everything else lives in the run-state file ($LOG_DIR/state.json) --
    local _L="$LOG_DIR"
    ss RUN_MODE               "$MODE"
    ss RUN_KIND               "$RUN_KIND"
    ss WORKTREE_DIR           "$wt"
    ss WORKTREE_BRANCH        "$WORKTREE_BRANCH"
    ss TEST_BACKEND           "$TEST_BACKEND"
    ss CREATE_LOCAL_BRANCH    "$CREATE_LOCAL_BRANCH"
    ss CREATE_PR              "$CREATE_PR"
    ss ISSUE_NUMBER           "$ISSUE_NUMBER"
    ss ISSUE_TITLE            "$ISSUE_TITLE"
    ss ISSUE_BODY             "$ISSUE_BODY"
    ss ISSUE_LABELS           "$ISSUE_LABELS"
    ss ISSUE_COMMENTS         "$ISSUE_COMMENTS"
    ss ISSUE_URL              "$(python "$S/state.py" --worktree-dir "$wt" get ISSUE_URL)"
    ss DASHBOARD_PROJECT_ID   "$DASHBOARD_PROJECT_ID"
    ss CODEGEN_LOGS_ROOT      "$CODEGEN_LOGS_ROOT"
    ss LOGS_BASE              "$LOGS_BASE"
    ss PR_REVIEW_KNOWLEDGE_DIR "$PR_REVIEW_KNOWLEDGE_DIR"
    ss START_TIME            "$START_TIME"
    ss RUN_ID                "$RUN_ID"
    ss LOG_DIR               "$LOG_DIR"
    ss GIT_COMMIT            "$GIT_COMMIT"
    ss GIT_BRANCH            "$GIT_BRANCH"
    ss CODEGEN_VERSION       "$CODEGEN_VERSION"
    ss TARGET_ARCHES_JSON    "$TARGET_ARCHES_JSON" --json
    ss ARCH_COUNT            "$ARCH_COUNT" --json
    ss ARCH_PROFILES_JSON    "$ARCH_PROFILES_JSON" --json
    [ "$MODE" = "single" ] && ss TARGET_ARCH "$TARGET_ARCH"
    # Counters + limits.
    ss COMPILATION_ATTEMPTS  0 --json
    ss DEBUG_CYCLES          0 --json
    ss MAX_DEBUG_CYCLES      5 --json
    ss TESTS_TOTAL           0 --json
    ss TESTS_PASSED          0 --json
    ss PERF_RETRIES          0 --json
    ss MAX_PERF_RETRIES      2 --json
    ss REVIEW_RETRIES        0 --json
    ss MAX_REVIEW_RETRIES    2 --json
    ss PERF_GOAL             "$PERF_GOAL"
    if [ "$RUN_KIND" = "review" ]; then ss PREVIOUS_AGENT "addresser"; else ss PREVIOUS_AGENT "analyzer"; fi
    ss VERIFY_DEFERRED       0 --json
    ss OBSTACLE              ""
    ss STATUS               ""
    ss FINAL_RESULT         ""

    # Session identity (captured while this is the most recently started session).
    local _SP SID PCWD
    _SP="$(python "$S/session_cost.py" --print-session 2>/dev/null || echo "")"
    SID="$(echo "$_SP" | awk '{print $1}')"; PCWD="$(echo "$_SP" | cut -d' ' -f2-)"
    if [ -n "$SID" ]; then ss SESSION_ID "$SID"; ss PROJECT_CWD "$PCWD"; fi

    echo "LOG_DIR=$LOG_DIR RUN_ID=$RUN_ID RUN_MODE=$MODE ARCHES=$TARGET_ARCHES_JSON"
}

# ===========================================================================
# Step 0 — write the initial run.json (pipeline steps + issue identity; multi
# seeds pending per-arch arch_results) and take the first cost snapshot.
# ===========================================================================
execute_step_write_initial_run_json() {
    local _L; _L="$(_LOG)"
    local S="$_ORCH_SCRIPTS" mode kind num title arches steps issue_json first_step
    mode="$(sg RUN_MODE)"; kind="$(sg RUN_KIND)"; num="$(sg ISSUE_NUMBER)"; title="$(sg ISSUE_TITLE)"
    arches="$(sg TARGET_ARCHES_JSON)"
    [ "$mode" = "multi" ] && steps="$_PIPELINE_STEPS_MULTI" || steps="$_PIPELINE_STEPS_SINGLE"
    first_step="analyzer"
    if [ "$kind" = "review" ]; then steps="$_PIPELINE_STEPS_REVIEW"; first_step="addresser"; fi

    issue_json="$(python - "$num" "$title" "$(sg ISSUE_URL)" "$(sg ISSUE_LABELS)" <<'PY'
import json, sys
num, title, url, labels = sys.argv[1:5]
print(json.dumps({
    "number": int(num),
    "title": title,
    "url": url or f"https://github.com/tenstorrent/tt-metal/issues/{num}",
    "labels": [l for l in labels.split(",") if l] if labels else [],
}))
PY
)"

    local prompt
    if [ "$kind" = "review" ]; then
        prompt="Address review comments on PR #$(sg PR_NUMBER) for issue #${num} using $(sg TEST_BACKEND) tests"
    elif [ "$mode" = "multi" ]; then prompt="Fix multi-arch issue #${num} using $(sg TEST_BACKEND) tests"
    else prompt="Fix $(sg TARGET_ARCH) issue #${num} using $(sg TEST_BACKEND) tests"; fi
    # rj() prepends `--log-dir "$_L"`, so it must NOT appear in this array.
    local -a common=(
        init --run-id "$(sg RUN_ID)"
        --kernel "issue_${num}" --kernel-type "issue_solver"
        --start-time "$(sg START_TIME)" --first-step "$first_step"
        --prompt "$prompt"
        --batch-id "${CODEGEN_BATCH_ID:-}" --model "${CODEGEN_MODEL:-sonnet}"
        --run-type "${CODEGEN_RUN_TYPE:-manual}"
        --git-commit "$(sg GIT_COMMIT)" --git-branch "$(sg GIT_BRANCH)"
        --version "$(sg CODEGEN_VERSION)" --description "#${num}: ${title}"
        --pipeline-steps "$steps" --issue "$issue_json"
    )

    local first_msg="Analyzing issue #${num}: ${title}"
    local first_msg_multi="Analyzing issue #${num} for ${arches}"
    if [ "$kind" = "review" ]; then
        first_msg="Addressing review comments on PR #$(sg PR_NUMBER)"
        first_msg_multi="$first_msg across ${arches}"
    fi

    if [ "$mode" = "multi" ]; then
        local init_patch
        init_patch="$(python - "$arches" "$(sg TEST_BACKEND)" "$(sg CREATE_LOCAL_BRANCH)" "$(sg CREATE_PR)" <<'PY'
import json, sys
arches = json.loads(sys.argv[1]); tb, clb, cpr = sys.argv[2:5]
print(json.dumps({
    "multi_arch_run": True,
    "target_arches": arches,
    "combined_status": "running",
    "arch_results": {a: {"status": "pending", "verdict": None, "tests_total": 0,
                         "tests_passed": 0, "obstacle": None} for a in arches},
    "test_backend": tb, "create_local_branch_requested": clb, "create_pr_requested": cpr,
}))
PY
)"
        rj "${common[@]}" --arch "multi" \
            --first-message "$first_msg_multi" \
            --phases-total "$(sg ARCH_COUNT)" --patch-json "$init_patch" || return $?
    else
        rj "${common[@]}" --arch "$(sg TARGET_ARCH)" \
            --first-message "$first_msg" || return $?
    fi
    # The PR being updated + the solve this descends from, for the dashboard.
    if [ "$kind" = "review" ]; then
        rj metric --patch-json "$(python - "$(sg PR_NUMBER)" \
            "$(sg SOURCE_RUN_ID)" "$(sg PR_HEAD_SHA)" <<'PY'
import json, sys
pr, src, head = sys.argv[1:4]
print(json.dumps({"run_kind": "review", "pr_number": int(pr or 0) or None,
                  "source_run_id": src or None, "pr_head_sha": head or None}))
PY
)" || return $?
    fi
    refresh_cost
}

# ===========================================================================
# Step 1 — refine PERF_GOAL from the analyzer's perf_intent line (optimize →
# improve, maintain → no_regress). No-op if the line is absent.
# ===========================================================================
execute_step_refine_perf_goal() {
    local _L; _L="$(_LOG)"
    local num pi; num="$(sg ISSUE_NUMBER)"
    pi="$(grep -ioE 'perf_intent:[[:space:]]*(optimize|maintain)' \
        "codegen/artifacts/issue_${num}_analysis.md" 2>/dev/null | head -1 | grep -ioE 'optimize|maintain')"
    case "$pi" in
        optimize) ss PERF_GOAL improve ;;
        maintain) ss PERF_GOAL no_regress ;;
    esac
    echo "PERF_GOAL=$(sg PERF_GOAL)"
}

# ===========================================================================
# Step 1.5 — normalize and seal required verification, then route from that
# immutable contract. The Markdown parser lives in run_json_writer.py; this
# shell step retains only the compatibility state consumed by existing agents.
# Required coverage, paths and selectors fail closed before a tester can start.
# ===========================================================================
execute_step_route_verification() {
    local _L; _L="$(_LOG)"
    local route_mode="${1:-normal}"
    local num A P M; num="$(sg ISSUE_NUMBER)"
    A="codegen/artifacts/issue_${num}_analysis.md"
    P="codegen/artifacts/issue_${num}_fix_plan.md"
    M="$_L/required_verification_manifest.json"
    local FIX_LAYER VERIFY_REQUIRED VERIFIABLE LLK_COVERAGE
    local METAL_TARGET METAL_COVERAGE METAL_FILTER METAL_DISPATCH ROUTE out
    gval() {
        grep -ioE "$1:[[:space:]]*[A-Za-z_]+" "$A" 2>/dev/null |
            head -1 | sed -E "s/.*:[[:space:]]*//" || true
    }
    mval() {
        sed -n '/^metal_verification:/,/^## /p' "$A" 2>/dev/null |
            grep -ioE "^[[:space:]]*$1:[[:space:]]*[A-Za-z_]+" |
            head -1 | sed -E "s/.*:[[:space:]]*//" || true
    }
    FIX_LAYER="$(gval 'fix_layer')"
    VERIFY_REQUIRED="$(gval 'verification_required')"
    VERIFIABLE="$(gval 'verifiable_in_llk_suite')"
    LLK_COVERAGE="$(gval 'llk_coverage')"
    METAL_TARGET="$(mval 'target')"
    METAL_COVERAGE="$(mval 'coverage')"
    METAL_FILTER="$(
        sed -n '/^metal_verification:/,/^## /p' "$A" 2>/dev/null |
            grep -ioE '^[[:space:]]*gtest_filter:.*' |
            head -1 | sed -E "s/^[[:space:]]*gtest_filter:[[:space:]]*//; s/^['\"]//; s/['\"]$//" || true
    )"
    METAL_DISPATCH="$(mval 'dispatch')"

    ss FIX_LAYER      "$FIX_LAYER"
    ss VERIFIABLE_IN_LLK "$VERIFIABLE"
    ss LLK_COVERAGE   "$LLK_COVERAGE"
    ss METAL_TARGET   "$METAL_TARGET"
    ss METAL_COVERAGE "$METAL_COVERAGE"
    ss METAL_FILTER   "$METAL_FILTER"
    ss METAL_DISPATCH "$METAL_DISPATCH"

    local -a manifest_args=(
        required-verification
        --output "$M"
        --analysis "$A"
        --plan "$P"
        --worktree "$(_wt)"
        --run-id "$(sg RUN_ID)"
        --expected-base-sha "$(sg GIT_COMMIT)"
        --architectures-json "$(sg TARGET_ARCHES_JSON)"
        --backend "$(sg TEST_BACKEND)"
    )
    if [ "$route_mode" = "hypothesis_refuted" ]; then
        manifest_args+=(--performance-only)
    elif [ "$route_mode" != "normal" ]; then
        echo "unsupported route verification mode: $route_mode" >&2
        return 1
    fi
    if [ -n "${CODEGEN_VERIFICATION_WAIVER_POLICY:-}" ]; then
        manifest_args+=(--waiver-policy "$CODEGEN_VERIFICATION_WAIVER_POLICY")
    fi
    if [ -f "$M" ]; then
        manifest_args+=(--supersedes-reason \
            "verification plan resealed after retry; debug=$(sg DEBUG_CYCLES), review=$(sg REVIEW_RETRIES), perf=$(sg PERF_RETRIES)")
    fi
    if ! out="$(rj "${manifest_args[@]}")"; then
        ROUTE=missing
        [ -n "$VERIFY_REQUIRED" ] || VERIFY_REQUIRED=yes
        ss VERIFY_REQUIRED "$VERIFY_REQUIRED"
        ss VERIFY_ROUTE "$ROUTE"
        ss REQUIRED_VERIFICATION_MANIFEST ""
        ss REQUIRED_VERIFICATION_MANIFEST_ID ""
        ss REQUIRED_VERIFICATION_ATTEMPT_ID ""
        rj message --message "Verify route rejected before execution: ${out:-required-verification manifest invalid}"
        printf '%s\n' "$out" >&2
        echo "VERIFY_ROUTE=missing REQUIRED_VERIFICATION=invalid"
        return 0
    fi

    local normalized
    normalized="$(python - "$M" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
suites = {r["suite"] for r in d["requirements"]}
functional = suites & {"llk", "metal"}
route = "both" if functional == {"llk", "metal"} else next(iter(functional), "none")
print(route, d["attempt_id"], d["manifest_id"], len(d["requirements"]))
PY
)"
    read -r ROUTE manifest_attempt manifest_id manifest_count <<<"$normalized"
    [ -n "$VERIFY_REQUIRED" ] || {
        if [ "$ROUTE" = none ]; then VERIFY_REQUIRED=no; else VERIFY_REQUIRED=yes; fi
    }
    if [ -z "$LLK_COVERAGE" ] && { [ "$ROUTE" = llk ] || [ "$ROUTE" = both ]; }; then
        LLK_COVERAGE=existing
        ss LLK_COVERAGE "$LLK_COVERAGE"
    fi
    ss VERIFY_REQUIRED "$VERIFY_REQUIRED"
    ss VERIFY_ROUTE   "$ROUTE"
    ss REQUIRED_VERIFICATION_MANIFEST "$M"
    ss REQUIRED_VERIFICATION_MANIFEST_ID "$manifest_id"
    ss REQUIRED_VERIFICATION_ATTEMPT_ID "$manifest_attempt"
    rj metric --patch-json "{\"required_verification\":{\"manifest_id\":\"${manifest_id}\",\"attempt_id\":\"${manifest_attempt}\",\"requirements\":${manifest_count}}}"
    rj message --message "Verify route: ${ROUTE}; sealed ${manifest_count} requirement(s) as ${manifest_id}"
    echo "$out"
    echo "VERIFY_ROUTE=$ROUTE MANIFEST_ID=$manifest_id ATTEMPT_ID=$manifest_attempt REQUIREMENTS=$manifest_count"
}

# ===========================================================================
# Step 2 — advance to arch_lookup (research). Sets PREVIOUS_AGENT=arch_lookup.
# ===========================================================================
execute_step_advance_arch_lookup() {
    local _L; _L="$(_LOG)"
    local num; num="$(sg ISSUE_NUMBER)"
    rj advance --new-step "arch_lookup" \
        --new-message "Researching architecture details for issue #${num}" \
        --prev-result "success" --prev-message "Issue analysis complete" --agent "analyzer"
    ss PREVIOUS_AGENT "arch_lookup"
}

# ===========================================================================
# Step 3 — advance to the writer (fix). Uses PREVIOUS_AGENT (analyzer|arch_lookup).
# ===========================================================================
execute_step_advance_writer() {
    local _L; _L="$(_LOG)"
    local num mode prev; num="$(sg ISSUE_NUMBER)"; mode="$(sg RUN_MODE)"; prev="$(sg PREVIOUS_AGENT)"
    if [ "$(sg STATUS)" = "skipped" ]; then
        echo "cannot advance a skipped run to writer" >&2
        return 1
    fi
    local msg="Planning and applying a fix for issue #${num}"
    [ "$mode" = multi ] && msg="Planning and applying one shared fix for issue #${num}"
    rj advance --new-step "writer" --new-message "$msg" \
        --prev-result "success" --prev-message "Analysis/research complete" --agent "${prev:-analyzer}"
}

# ===========================================================================
# Step 3 — record the worker's tracked and untracked changed files into state
# (for tester/reviewer).
# ===========================================================================
execute_step_record_changed_files() {
    local _L; _L="$(_LOG)"
    local wt tracked untracked cf test_changes; wt="$(_wt)"
    tracked="$(git -C "$wt" diff --name-only 2>/dev/null || true)"
    untracked="$(git -C "$wt" ls-files --others --exclude-standard 2>/dev/null || true)"
    cf="$(printf '%s\n%s\n' "$tracked" "$untracked" | sed '/^$/d' | sort -u)"
    ss CHANGED_FILES "$cf"
    test_changes="$(printf '%s\n' "$cf" | grep -E '(^|/)tests?/|(^|/)test_[^/]+$' || true)"
    [ -z "$test_changes" ] || rj metric --patch-json '{"tests_generated":true}'
    printf '%s\n' "$cf"
}

# ===========================================================================
# Step 4 — VERIFY_ROUTE=none: runtime verification is explicitly not applicable.
# Mark every in-scope arch UNVERIFIABLE_IN_LLK_SUITE (multi) and record the
# non-runtime outcome. Missing required coverage uses VERIFY_ROUTE=missing and
# must never reach this helper.
# ===========================================================================
execute_step_mark_unverifiable() {
    local _L; _L="$(_LOG)"
    local mode fl arch; mode="$(sg RUN_MODE)"; fl="$(sg FIX_LAYER)"
    if [ "$mode" = "multi" ]; then
        for arch in $(python -c "import json,sys;print(' '.join(json.load(sys.stdin)))" <<<"$(sg TARGET_ARCHES_JSON)"); do
            rj metric --patch-json "{\"arch_results\":{\"${arch}\":{\"status\":\"done\",\"verdict\":\"UNVERIFIABLE_IN_LLK_SUITE\",\"tests_total\":0,\"tests_passed\":0,\"obstacle\":null}}}"
        done
    fi
    ss VERIFY_DEFERRED 1 --json
    ss VERIFY_DEFER_NOTE "fix applied + committed; runtime verification is not applicable to this ${fl} change"
}

# ===========================================================================
# Step 4 — advance to the tt-llk tester. Optional arg: prev agent (default
# "writer"; pass "fix_tests" on a debug re-test).
# ===========================================================================
execute_step_advance_tester() {
    local _L; _L="$(_LOG)"
    local agent="${1:-writer}" num mode arches be
    num="$(sg ISSUE_NUMBER)"; mode="$(sg RUN_MODE)"; arches="$(sg TARGET_ARCHES_JSON)"; be="$(sg TEST_BACKEND)"
    local msg="Running ${be} tests for issue #${num}"
    [ "$mode" = multi ] && msg="Running ${be} tests for issue #${num} across ${arches}"
    rj advance --new-step "tester" --new-message "$msg" \
        --prev-result "success" --prev-message "Fix applied" --agent "$agent"
}

# ===========================================================================
# Step 4b — advance to the metal unit_tests_llk suite.
# ===========================================================================
execute_step_advance_metal_test() {
    local _L; _L="$(_LOG)"
    local num mode arches route filt; num="$(sg ISSUE_NUMBER)"; mode="$(sg RUN_MODE)"
    arches="$(sg TARGET_ARCHES_JSON)"; route="$(sg VERIFY_ROUTE)"; filt="$(sg METAL_FILTER)"
    local scope="issue #${num}"; [ "$mode" = multi ] && scope="issue #${num} across ${arches}"
    rj advance --new-step "metal_test" \
        --new-message "Building+running unit_tests_llk (${filt}) for ${scope}" \
        --prev-result "success" --prev-message "${route:-metal} route" --agent "writer"
}

# ===========================================================================
# Step 4 — combine the required suite results for each architecture. Testers
# write only arch_results.<arch>.suite_results.<llk|metal>; this function owns
# the compatibility verdict/count fields consumed by final status and the
# dashboard. Missing or unknown required results fail closed as ENV_ERROR.
# ===========================================================================
execute_step_combine_verification_results() {
    local _L; _L="$(_LOG)"
    local route arches patch pool manifest
    route="$(sg VERIFY_ROUTE)"
    arches="$(sg TARGET_ARCHES_JSON)"
    case "$route" in llk|metal|both) ;; *) echo "cannot combine VERIFY_ROUTE=$route" >&2; return 1 ;; esac

    pool="$(python - "$_L/run.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("runner_pool") or "prod")
PY
)" || return 1
    if [ "$pool" = audit ]; then
        manifest="$(sg REQUIRED_VERIFICATION_MANIFEST)"
        [ -f "$manifest" ] || {
            echo "audit verification manifest is missing: ${manifest:-unset}" >&2
            return 1
        }
        rj reduce-verification \
            --manifest "$manifest" \
            --results-dir "$_L/verification-results" \
            --scope functional \
            --worktree "$(_wt)" \
            --perf-result "$_L/perf_result.json" \
            --output "$_L/verification_reduction.json"
        return $?
    fi

    patch="$(python - "$_L/run.json" "$arches" "$route" <<'PY'
import json
import sys

run_path, arches_json, route = sys.argv[1:4]
with open(run_path) as f:
    run = json.load(f)

arches = json.loads(arches_json)
required = {"llk": ("llk",), "metal": ("metal",), "both": ("llk", "metal")}[route]
failure_priority = ("ENV_ERROR", "COMPILE_FAILED", "TESTS_FAILED", "SIM_ISA_GAP")
existing = run.get("arch_results") or {}
updates = {}
tests_total = 0
tests_passed = 0

def strict_count(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value

for arch in arches:
    current = existing.get(arch) or {}
    if current.get("verdict") == "SKIPPED":
        tests_total += strict_count(
            current.get("tests_total"), f"{arch}.tests_total"
        )
        tests_passed += strict_count(
            current.get("tests_passed"), f"{arch}.tests_passed"
        )
        continue

    suite_results = current.get("suite_results") or {}
    verdicts = []
    reasons = []
    arch_total = 0
    arch_passed = 0
    for suite_name in required:
        suite = suite_results.get(suite_name) or {}
        verdict = suite.get("verdict")
        suite_reasons = []
        if suite.get("status") != "done":
            suite_reasons.append("RESULT_NOT_TERMINAL: status must be done")
        try:
            suite_total = strict_count(
                suite.get("tests_total"), f"{suite_name}.tests_total"
            )
            suite_passed = strict_count(
                suite.get("tests_passed"), f"{suite_name}.tests_passed"
            )
            if suite_passed > suite_total:
                raise ValueError(f"{suite_name}.tests_passed exceeds tests_total")
        except ValueError as exc:
            suite_total = 0
            suite_passed = 0
            suite_reasons.append(f"COUNT_INVALID: {exc}")

        if not verdict:
            verdict = "ENV_ERROR"
            suite_reasons.append("RESULT_MISSING: missing required suite verdict")
        elif verdict == "SUCCESS":
            if suite_total < 1:
                suite_reasons.append("ZERO_SELECTED: success requires tests_total > 0")
            if suite_passed != suite_total:
                suite_reasons.append(
                    "COUNT_MISMATCH: success requires tests_passed == tests_total"
                )
        elif verdict not in failure_priority:
            suite_reasons.append(f"VERDICT_UNKNOWN: {verdict}")
            verdict = "ENV_ERROR"
        if suite_reasons:
            verdict = "ENV_ERROR"
            reasons.extend(f"{suite_name}: {reason}" for reason in suite_reasons)
        elif verdict in failure_priority:
            reasons.append(f"{suite_name}: {suite.get('obstacle') or verdict}")
        elif suite.get("obstacle"):
            reasons.append(f"{suite_name}: {suite['obstacle']}")
        verdicts.append(verdict)
        arch_total += suite_total
        arch_passed += suite_passed

    if verdicts and all(verdict == "SUCCESS" for verdict in verdicts):
        combined = "SUCCESS"
    else:
        combined = next(
            (value for value in failure_priority if value in verdicts), "ENV_ERROR"
        )

    updates[arch] = {
        "status": "done",
        "verdict": combined,
        "verification_route": route,
        "tests_total": arch_total,
        "tests_passed": arch_passed,
        "obstacle": "; ".join(reasons) or None,
    }
    tests_total += arch_total
    tests_passed += arch_passed

print(json.dumps({
    "arch_results": updates,
    "tests_total": tests_total,
    "tests_passed": tests_passed,
}))
PY
)" || return 1

    rj metric --patch-json "$patch"
}

# ===========================================================================
# Step 4 — roll the combined per-arch counters into state. arch_results is the
# source of truth in run.json; this keeps finalize independent of shell state
# that does not survive between Bash calls.
# ===========================================================================
execute_step_aggregate_results() {
    local _L; _L="$(_LOG)"
    local agg
    agg="$(python - "$_L/run.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("0 0"); raise SystemExit
ar = d.get("arch_results") or {}
if ar:
    tt = sum(int(v.get("tests_total") or 0) for v in ar.values())
    tp = sum(int(v.get("tests_passed") or 0) for v in ar.values())
else:
    tt = int(d.get("tests_total") or 0); tp = int(d.get("tests_passed") or 0)
print(f"{tt} {tp}")
PY
)"
    ss TESTS_TOTAL  "${agg%% *}" --json
    ss TESTS_PASSED "${agg##* }" --json
    echo "TESTS_TOTAL=${agg%% *} TESTS_PASSED=${agg##* }"
}

# ===========================================================================
# Debug/review/perf feedback — record a failure and advance to fix_tests.
# Args: <step> <agent> <summary> <retry_msg>.
# ===========================================================================
execute_step_feedback() {
    local _L; _L="$(_LOG)"
    local step="$1" agent="$2" summary="$3" retry_msg="$4"
    rj failure --step "$step" --agent "$agent" --type "test_failure" --message "$summary" --resolved "false"
    rj advance --new-step "fix_tests" --new-message "$retry_msg" \
        --prev-result "test_failure" --prev-message "$summary" --agent "$agent"
}

# Convenience wrappers over execute_step_feedback for the three loops.
execute_step_debug_feedback() {
    local _L; _L="$(_LOG)"
    local summary="$1" num dc mdc; num="$(sg ISSUE_NUMBER)"; dc="$(sg DEBUG_CYCLES)"; mdc="$(sg MAX_DEBUG_CYCLES)"
    execute_step_feedback "tester" "tester" "$summary" \
        "Debugging test failure for issue #${num} (attempt $((dc+1))/${mdc})"
}
execute_step_coverage_feedback() {
    local _L; _L="$(_LOG)"
    local summary="${1:-MISSING_TEST_COVERAGE: required runnable regression coverage was not added}"
    local num dc mdc; num="$(sg ISSUE_NUMBER)"; dc="$(sg DEBUG_CYCLES)"; mdc="$(sg MAX_DEBUG_CYCLES)"
    execute_step_feedback "writer" "writer" "$summary" \
        "Adding missing test coverage for issue #${num} (attempt $((dc+1))/${mdc})"
}
execute_step_review_feedback() {
    local _L; _L="$(_LOG)"
    local summary="$1" num rr mrr; num="$(sg ISSUE_NUMBER)"; rr="$(sg REVIEW_RETRIES)"; mrr="$(sg MAX_REVIEW_RETRIES)"
    execute_step_feedback "review" "reviewer" "$summary" \
        "Addressing review findings for issue #${num}; attempt $((rr+1))/${mrr}"
}
execute_step_perf_feedback() {
    local _L; _L="$(_LOG)"
    local summary="$1" num pr mpr goal; num="$(sg ISSUE_NUMBER)"; pr="$(sg PERF_RETRIES)"; mpr="$(sg MAX_PERF_RETRIES)"; goal="$(sg PERF_GOAL)"
    execute_step_feedback "perf" "perf" "$summary" \
        "Recovering perf for issue #${num} (${goal}); attempt $((pr+1))/${mpr}"
}
# RUN_KIND=review: the retry goes back to the addresser, not the issue worker.
execute_step_review_round_feedback() {
    local _L; _L="$(_LOG)"
    local step="${1:-tester}" summary="$2" prnum dc mdc
    prnum="$(sg PR_NUMBER)"; dc="$(sg DEBUG_CYCLES)"; mdc="$(sg MAX_DEBUG_CYCLES)"
    execute_step_feedback "$step" "$step" "$summary" \
        "Repairing the review fix on PR #${prnum} (attempt $((dc+1))/${mdc})"
}

# Counter bumps (called after the retry worker returns).
execute_step_bump_debug()  { local _L; _L="$(_LOG)"; ss DEBUG_CYCLES  "$(( $(sg DEBUG_CYCLES) + 1 ))" --json; echo "DEBUG_CYCLES=$(sg DEBUG_CYCLES)/$(sg MAX_DEBUG_CYCLES)"; }
execute_step_bump_review() { local _L; _L="$(_LOG)"; ss REVIEW_RETRIES "$(( $(sg REVIEW_RETRIES) + 1 ))" --json; ss DEBUG_CYCLES "$(( $(sg DEBUG_CYCLES) + 1 ))" --json; echo "REVIEW_RETRIES=$(sg REVIEW_RETRIES)/$(sg MAX_REVIEW_RETRIES)"; }
execute_step_bump_perf()   { local _L; _L="$(_LOG)"; ss PERF_RETRIES "$(( $(sg PERF_RETRIES) + 1 ))" --json; ss DEBUG_CYCLES "$(( $(sg DEBUG_CYCLES) + 1 ))" --json; echo "PERF_RETRIES=$(sg PERF_RETRIES)/$(sg MAX_PERF_RETRIES)"; }

# ===========================================================================
# Step 5.3 — advance to the reviewer.
# ===========================================================================
execute_step_advance_review() {
    local _L; _L="$(_LOG)"
    local num mode arches rr mrr; num="$(sg ISSUE_NUMBER)"; mode="$(sg RUN_MODE)"
    arches="$(sg TARGET_ARCHES_JSON)"; rr="$(sg REVIEW_RETRIES)"; mrr="$(sg MAX_REVIEW_RETRIES)"
    local what="fix diff for issue #${num}"
    [ "$mode" = multi ] && what="shared fix diff for issue #${num} across ${arches}"
    rj advance --new-step "review" \
        --new-message "Reviewing ${what} (attempt $((rr+1))/$((mrr+1)))" \
        --prev-result "success" --prev-message "Functional tests passed" --agent "tester"
}

# ===========================================================================
# Step 5.3 — patch the reviewer's result file into run.json.
# ===========================================================================
execute_step_record_review() {
    local _L; _L="$(_LOG)"
    [ -f "$_L/review_result.json" ] || { echo "no review_result.json to record"; return 0; }
    rj metric --patch-json "{\"review\": $(cat "$_L/review_result.json")}"
}

# ===========================================================================
# Step 5.5 — compute and store the perf-eligible arches (local Blackhole/
# Wormhole only). Prints them space-separated (empty → skip perf).
# ===========================================================================
execute_step_perf_arches() {
    local _L; _L="$(_LOG)"
    local arches be keep; arches="$(sg TARGET_ARCHES_JSON)"; be="$(sg TEST_BACKEND)"
    # A review round measures perf only when a disposition asked for it.
    if [ "$(sg RUN_KIND)" = "review" ] && [ "$(sg PERF_REQUESTED)" != "1" ]; then
        ss PERF_ARCHES ""
        echo ""
        return 0
    fi
    keep="$(python - "$arches" "$be" <<'PY'
import json, sys
arches = json.loads(sys.argv[1]); backend = sys.argv[2]
print(" ".join(a for a in arches if backend == "local" and a in ("blackhole", "wormhole")))
PY
)"
    ss PERF_ARCHES "$keep"
    echo "$keep"
}

# ===========================================================================
# Step 5.5 — record perf as not measured for the run.
# ===========================================================================
execute_step_perf_not_measured() {
    local _L; _L="$(_LOG)"
    rj metric --patch-json "{\"perf\": {\"measured\": false, \"verdict\": \"not_measured\", \"reason\": \"perf only runs on local Blackhole/Wormhole silicon\"}}"
}

# ===========================================================================
# Step 5.5 — advance to the perf step.
# ===========================================================================
execute_step_advance_perf() {
    local _L; _L="$(_LOG)"
    local num mode goal pa; num="$(sg ISSUE_NUMBER)"; mode="$(sg RUN_MODE)"; goal="$(sg PERF_GOAL)"; pa="$(sg PERF_ARCHES)"
    local where="${pa:-$(sg TARGET_ARCH)}"
    rj advance --new-step "perf" \
        --new-message "Measuring perf for issue #${num} on ${where} (goal=${goal})" \
        --prev-result "success" --prev-message "Functional tests passed" --agent "perf"
}

# ===========================================================================
# Step 5.5 — record the perf-tester's result. Multi: under arch_results.<arch>.
# Single: top-level perf. Reads $LOG_DIR/perf_result.json. Arg (multi): <arch>.
# ===========================================================================
execute_step_record_perf() {
    local _L; _L="$(_LOG)"
    local mode arch; mode="$(sg RUN_MODE)"; arch="$1"
    [ -f "$_L/perf_result.json" ] || { echo "no perf_result.json to record"; return 0; }
    if [ "$mode" = "multi" ]; then
        rj metric --patch-json "{\"arch_results\": {\"${arch}\": {\"perf\": $(cat "$_L/perf_result.json")}}}"
    else
        rj metric --patch-json "{\"perf\": $(cat "$_L/perf_result.json")}"
    fi
}

# ===========================================================================
# Set the terminal STATUS/FINAL_RESULT/SOLVER_STATE. Args: <status> [final_result].
# final_result defaults per status (success/compiled/skipped → success, else test_failure).
# ===========================================================================
execute_step_mark_status() {
    local _L; _L="$(_LOG)"
    local status="$1" fr="$2"
    if [ -z "$fr" ]; then
        case "$status" in success|compiled|skipped) fr=success ;; *) fr=test_failure ;; esac
    fi
    ss STATUS "$status"
    ss FINAL_RESULT "$fr"
    case "$status" in success|compiled) ss SOLVER_STATE working ;; *) ss SOLVER_STATE not_working ;; esac
    echo "STATUS=$status FINAL_RESULT=$fr"
}

# ===========================================================================
# Analyzer terminal path — finalize immediately when the whole issue is out of
# LLK scope. No writer, verification, review, performance, or fix packaging.
# ===========================================================================
execute_step_finalize_out_of_scope() {
    local _L; _L="$(_LOG)"
    local mode num analysis msg arches patch
    mode="$(sg RUN_MODE)"; num="$(sg ISSUE_NUMBER)"
    analysis="codegen/artifacts/issue_${num}_analysis.md"

    if [ ! -f "$analysis" ] || ! grep -Eq \
        '^[[:space:]]*in_scope:[[:space:]]*false([[:space:]]|$)' "$analysis"; then
        echo "cannot skip: analyzer did not mark the whole issue out of scope" >&2
        return 1
    fi

    ss OBSTACLE ""
    ss BASE_COMMIT "$(sg GIT_COMMIT)"
    ss FIX_COMMIT ""
    ss CHANGED_FILES ""
    ss CHANGED_FILES_JSON '[]' --json
    rm -f "$_L/generated.patch"

    if [ "$mode" = "multi" ]; then
        arches="$(sg TARGET_ARCHES_JSON)"
        patch="$(python - "$arches" <<'PY'
import json
import sys

arches = json.loads(sys.argv[1])
print(json.dumps({
    "combined_status": "skipped",
    "arch_results": {
        arch: {
            "status": "done",
            "verdict": "SKIPPED",
            "tests_total": 0,
            "tests_passed": 0,
            "obstacle": None,
        }
        for arch in arches
    },
}))
PY
)"
        rj metric --patch-json "$patch" || return $?
        ss COMBINED_STATUS skipped
        msg="multi-arch issue #${num}: skipped — outside LLK scope"
    else
        msg="$(sg TARGET_ARCH) issue #${num}: skipped — outside LLK scope"
    fi

    execute_step_mark_status skipped
    ss FINAL_MESSAGE "$msg"
    execute_step_finalize_run || return $?
    execute_step_copy_artifacts
}

# ===========================================================================
# Step 6 (single) — map the tester/metal verdict to the terminal status.
# Arg: <verdict>. Does not override an already-set failed STATUS (loop exhaustion).
# ===========================================================================
execute_step_status_from_verdict() {
    local _L; _L="$(_LOG)"
    local verdict="$1" cur; cur="$(sg STATUS)"
    if [ "$cur" = "failed" ]; then echo "STATUS already failed (kept)"; return 0; fi
    case "$verdict" in
        SUCCESS)                                   execute_step_mark_status success ;;
        COMPILED_ONLY|UNVERIFIABLE_IN_LLK_SUITE)   execute_step_mark_status compiled ;;
        SKIPPED)                                   execute_step_mark_status skipped ;;
        COMPILE_FAILED)                            execute_step_mark_status failed compile_error ;;
        *)                                         execute_step_mark_status failed test_failure ;;
    esac
}

# ===========================================================================
# Step 6 (multi) — compute combined_status from the per-arch arch_results in
# run.json (the decision table, in code), store COMBINED_STATUS, and set the
# terminal STATUS/FINAL_RESULT. Does not override an already-set failed STATUS.
# ===========================================================================
execute_step_combined_status() {
    local _L; _L="$(_LOG)"
    local cur; cur="$(sg STATUS)"
    local out combined status
    out="$(python - "$_L/run.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
ar = d.get("arch_results") or {}
PASS_REAL = {"SUCCESS"}
PASS_COMPILED = {"COMPILED_ONLY"}
UNVERIFIED = {"UNVERIFIABLE_IN_LLK_SUITE"}
FAILED = {"COMPILE_FAILED", "TESTS_FAILED", "ENV_ERROR", "SIM_ISA_GAP"}
in_scope = {a: v for a, v in ar.items() if (v.get("verdict") or "") != "SKIPPED"}
verdicts = [(v.get("verdict") or "") for v in in_scope.values()]
n = len(in_scope)
any_real = any(x in PASS_REAL for x in verdicts)
any_failed = any(x in FAILED for x in verdicts)
all_passed = n > 0 and all(x in (PASS_REAL | PASS_COMPILED | UNVERIFIED) for x in verdicts)
all_failed = n > 0 and all(x in FAILED for x in verdicts)
if n == 0:
    combined = "skipped"                                   # every arch out of scope
elif all_passed and any_real:
    combined = "success"                                   # ≥1 verified by a real test
elif all_passed:
    combined = "compiled"                                  # fixed, none failed, no real test
elif any_failed and (any_real or any(x in (PASS_COMPILED | UNVERIFIED) for x in verdicts)):
    combined = "partial"
elif all_failed:
    combined = "failed"
else:
    combined = "partial"
status = {"success": "success", "compiled": "compiled", "skipped": "skipped",
          "partial": "failed", "failed": "failed"}[combined]
print(f"{combined}\t{status}")
PY
)"
    combined="${out%%$'\t'*}"; status="${out##*$'\t'}"
    ss COMBINED_STATUS "$combined"
    rj metric --patch-json "{\"combined_status\": \"${combined}\"}"
    if [ "$cur" = "failed" ]; then echo "STATUS already failed (kept); combined_status=${combined}"; return 0; fi
    local fr; case "$status" in success|compiled|skipped) fr=success ;; *) fr=test_failure ;; esac
    execute_step_mark_status "$status" "$fr"
    echo "COMBINED_STATUS=$combined STATUS=$status"
}

# ===========================================================================
# Step 6 — deferred-verification messaging (VERIFY_ROUTE=none / VERIFY_DEFERRED).
# Clears OBSTACLE (Working outcome) and sets a next-step FINAL_MESSAGE.
# ===========================================================================
execute_step_deferred_message() {
    local _L; _L="$(_LOG)"
    [ "$(sg VERIFY_DEFERRED)" = "1" ] || return 0
    local mode num fl note tag; mode="$(sg RUN_MODE)"; num="$(sg ISSUE_NUMBER)"; fl="$(sg FIX_LAYER)"
    note="$(sg VERIFY_DEFER_NOTE)"; note="${note:-runtime verification is not applicable to this ${fl} change}"
    tag="$(sg TARGET_ARCH)"; [ "$mode" = multi ] && tag="multi-arch"
    ss OBSTACLE ""
    ss FINAL_MESSAGE "${tag} issue #${num}: fix applied — ${note}"
}

# ===========================================================================
# Step 6 — local commit of the fix + generated.patch. Computes CHANGED_FILES,
# BASE_COMMIT, FIX_COMMIT into state. No push (caller owns push/PR).
# ===========================================================================
execute_step_write_generated_patch() {
    local _L; _L="$(_LOG)"
    local wt num title; wt="$(_wt)"; num="$(sg ISSUE_NUMBER)"; title="$(sg ISSUE_TITLE)"
    local mode; mode="$(sg RUN_MODE)"
    local cf cfj base fix packaged tmp_patch
    # Input validation requires a clean dedicated worktree, so every non-ignored
    # change now belongs to this run. Stage the whole worktree and exclude only
    # generated test infrastructure.
    local -a pathspec=(
        .
        ':(exclude,glob)**/perf_data/**'
        ':(exclude,glob)**/__pycache__/**'
        ':(exclude)tt_metal/tt-llk/tests/.venv'
        ':(exclude,glob)tt_metal/tt-llk/tests/.venv/**'
        ':(exclude)tt_metal/tt-llk/tests/sfpi'
        ':(exclude,glob)tt_metal/tt-llk/tests/sfpi/**'
    )

    base="$(sg GIT_COMMIT)"
    if ! git -C "$wt" rev-parse --verify "${base}^{commit}" >/dev/null 2>&1; then
        ss OBSTACLE "packaging failed: invalid base commit"
        execute_step_mark_status failed
        echo "PACKAGING_FAILED: invalid base commit '$base'" >&2
        return 1
    fi
    ss BASE_COMMIT "$base"

    if ! git -C "$wt" -c advice.addIgnoredFile=false add -A -- "${pathspec[@]}"; then
        ss OBSTACLE "packaging failed: could not stage the complete fix"
        execute_step_mark_status failed
        echo "PACKAGING_FAILED: git add failed" >&2
        return 1
    fi

    cf="$(git -C "$wt" diff --cached --name-only)"
    ss CHANGED_FILES "$cf"
    cfj="$(CF="$cf" python -c "import json,os;print(json.dumps([l for l in os.environ['CF'].splitlines() if l]))")"
    ss CHANGED_FILES_JSON "$cfj" --json

    fix=""
    if ! git -C "$wt" diff --cached --quiet 2>/dev/null; then
        local cm="AI issue-solver: fix #${num} ${title}"
        [ "$mode" = multi ] && cm="AI issue-solver: multi-arch fix #${num} ${title}"
        if ! git -C "$wt" -c user.name="ai-code-gen" -c user.email="ai-code-gen@tenstorrent.com" \
            commit -q -m "$cm"; then
            ss OBSTACLE "packaging failed: could not commit the complete fix"
            execute_step_mark_status failed
            echo "PACKAGING_FAILED: git commit failed; fix remains staged" >&2
            return 1
        fi
        fix="$(git -C "$wt" rev-parse HEAD)"
    fi
    ss FIX_COMMIT "$fix"

    if [ -n "$fix" ] && [ "$fix" != "$base" ]; then
        packaged="$(git -C "$wt" diff --name-only "$base" "$fix")"
        if [ "$packaged" != "$cf" ]; then
            ss OBSTACLE "packaging failed: committed paths do not match the staged fix"
            execute_step_mark_status failed
            echo "PACKAGING_FAILED: staged and committed path sets differ" >&2
            return 1
        fi

        tmp_patch="$_L/.generated.patch.$$"
        if ! git -C "$wt" diff --binary "$base" "$fix" > "$tmp_patch"; then
            rm -f "$tmp_patch"
            ss OBSTACLE "packaging failed: could not create generated.patch"
            execute_step_mark_status failed
            echo "PACKAGING_FAILED: git diff failed" >&2
            return 1
        fi
        if [ ! -s "$tmp_patch" ]; then
            rm -f "$tmp_patch"
            ss OBSTACLE "packaging failed: generated.patch is empty"
            execute_step_mark_status failed
            echo "PACKAGING_FAILED: generated.patch is empty" >&2
            return 1
        fi
        if ! _disk_guard mv "$tmp_patch" "$_L/generated.patch"; then
            rm -f "$tmp_patch"
            ss OBSTACLE "packaging failed: could not publish generated.patch"
            execute_step_mark_status failed
            echo "PACKAGING_FAILED: could not publish generated.patch" >&2
            return 1
        fi
    else
        rm -f "$_L/generated.patch"
    fi

    echo "FIX_COMMIT=${fix:-none} CHANGED=$(printf '%s' "$cf" | grep -c . || true)"
}

# ===========================================================================
# Step 6 — finalize run.json. Reads terminal state (STATUS set by
# combined_status / status_from_verdict / a loop-exhaustion mark_status),
# writes the finalize record, and does the authoritative cost refresh.
# ===========================================================================
execute_step_finalize_run() {
    local _L; _L="$(_LOG)"
    local S="$_ORCH_SCRIPTS" mode num status fr ss_state end pool manifest reduction_class
    mode="$(sg RUN_MODE)"; num="$(sg ISSUE_NUMBER)"

    pool="$(python - "$_L/run.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("runner_pool") or "prod")
PY
)" || return 1
    if [ "$pool" = audit ]; then
        manifest="$(sg REQUIRED_VERIFICATION_MANIFEST)"
        if [ ! -f "$manifest" ]; then
            ss OBSTACLE "audit verification manifest is missing"
            ss FINAL_MESSAGE "audit failed: required-verification manifest is missing"
            execute_step_mark_status failed test_failure
        elif ! rj reduce-verification \
                --manifest "$manifest" \
                --results-dir "$_L/verification-results" \
                --scope all \
                --worktree "$(_wt)" \
                --perf-result "$_L/perf_result.json" \
                --output "$_L/verification_reduction.json"; then
            ss OBSTACLE "verification reduction could not validate its inputs"
            ss FINAL_MESSAGE "audit failed: verification reduction inputs are invalid"
            execute_step_mark_status failed test_failure
        else
            reduction_class="$(python - "$_L/verification_reduction.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("classification") or "invalid")
PY
)" || reduction_class=invalid
            if [ "$(sg STATUS)" = success ] && [ "$reduction_class" != success ]; then
                ss OBSTACLE "verification reduction rejected success: ${reduction_class}"
                ss FINAL_MESSAGE "audit failed: verification reduction is ${reduction_class}"
                execute_step_mark_status failed test_failure
            fi
        fi
    fi

    status="$(sg STATUS)"; fr="$(sg FINAL_RESULT)"; ss_state="$(sg SOLVER_STATE)"
    end="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    local fmsg; fmsg="$(sg FINAL_MESSAGE)"
    if [ -z "$fmsg" ]; then
        if [ "$mode" = multi ]; then fmsg="multi-arch issue #${num}: $(sg COMBINED_STATUS)"; else fmsg="$(sg TARGET_ARCH) issue #${num}: ${status}"; fi
    fi

    local patch
    patch="$(LOG_DIR="$_L" \
        COMPILATION_ATTEMPTS="$(sg COMPILATION_ATTEMPTS)" DEBUG_CYCLES="$(sg DEBUG_CYCLES)" \
        TESTS_TOTAL="$(sg TESTS_TOTAL)" TESTS_PASSED="$(sg TESTS_PASSED)" \
        CHANGED_FILES_JSON="$(sg CHANGED_FILES_JSON)" TEST_BACKEND="$(sg TEST_BACKEND)" \
        CREATE_LOCAL_BRANCH="$(sg CREATE_LOCAL_BRANCH)" CREATE_PR="$(sg CREATE_PR)" \
        BASE_COMMIT="$(sg BASE_COMMIT)" FIX_COMMIT="$(sg FIX_COMMIT)" \
        GIT_BRANCH="$(sg GIT_BRANCH)" WORKTREE_BRANCH="$(sg WORKTREE_BRANCH)" \
        WORKTREE_DIR="$(sg WORKTREE_DIR)" OBSTACLE="$(sg OBSTACLE)" \
        RUN_MODE="$mode" TARGET_ARCHES_JSON="$(sg TARGET_ARCHES_JSON)" COMBINED_STATUS="$(sg COMBINED_STATUS)" \
        python - <<'PY'
import json, os
log_dir = os.environ["LOG_DIR"]
run_path = os.path.join(log_dir, "run.json")
try:
    run = json.load(open(run_path))
except FileNotFoundError:
    run = {}
agents = run.get("agents", [])
for agent, filename in [
    ("analyzer", "agent_issue_analyzer.md"),
    ("arch_lookup", "agent_arch_lookup.md"),
    ("addresser", "agent_review_addresser.md"),
    ("writer", "agent_issue_worker.md"),
    ("tester", "agent_tester.md"),
    ("metal_test", "agent_metal_tester.md"),
    ("reviewer", "agent_reviewer.md"),
    ("perf", "agent_perf_tester.md"),
    ("fix_tests", "agent_issue_worker_debug.md"),
]:
    if os.path.exists(os.path.join(log_dir, filename)) and agent not in agents:
        agents.append(agent)
patch = {
    "compilation_attempts": int(os.environ.get("COMPILATION_ATTEMPTS", "0") or 0),
    "debug_cycles": int(os.environ.get("DEBUG_CYCLES", "0") or 0),
    "tests_total": int(os.environ.get("TESTS_TOTAL", "0") or 0),
    "tests_passed": int(os.environ.get("TESTS_PASSED", "0") or 0),
    "agents": agents,
    "changed_files": json.loads(os.environ.get("CHANGED_FILES_JSON") or "[]"),
    "test_backend": os.environ.get("TEST_BACKEND", ""),
    "create_local_branch_requested": os.environ.get("CREATE_LOCAL_BRANCH", ""),
    "create_pr_requested": os.environ.get("CREATE_PR", ""),
    "base_commit": os.environ.get("BASE_COMMIT") or None,
    "fix_commit": os.environ.get("FIX_COMMIT") or None,
    "branch": os.environ.get("GIT_BRANCH") or os.environ.get("WORKTREE_BRANCH") or None,
    "worktree_dir": os.environ.get("WORKTREE_DIR") or None,
    "artifact_patch": "generated.patch" if os.path.exists(os.path.join(log_dir, "generated.patch")) else None,
    "obstacle": os.environ.get("OBSTACLE") or None,
}
if os.environ.get("RUN_MODE") == "multi":
    patch["multi_arch_run"] = True
    patch["target_arches"] = json.loads(os.environ.get("TARGET_ARCHES_JSON") or "[]")
    patch["combined_status"] = os.environ.get("COMBINED_STATUS") or None
    patch["arch_results"] = run.get("arch_results", {})
print(json.dumps(patch))
PY
)"

    rj finalize --end-time "$end" --status "$status" --final-result "$fr" \
        --worktree "$(_wt)" \
        --solver-state "$ss_state" --final-message "$fmsg" --patch-json "$patch" || return $?
    refresh_cost
    echo "finalized: status=$status final_result=$fr"
}

# ===========================================================================
# Step 6 — append the run to runs.jsonl and copy artifacts + base snapshots
# into LOG_DIR so the run is self-contained after the worktree is removed.
# ===========================================================================
execute_step_copy_artifacts() {
    local _L; _L="$(_LOG)"
    local S="$_ORCH_SCRIPTS" wt num lb; wt="$(_wt)"; num="$(sg ISSUE_NUMBER)"; lb="$(sg LOGS_BASE)"
    python "$S/issue_solver_run_utils.py" upsert-runs-jsonl --log-dir "$_L" --runs-jsonl "${lb}/runs.jsonl" || true
    cp codegen/artifacts/issue_${num}_*.md "$_L/" 2>/dev/null || true
    local f flat base; base="$(sg GIT_COMMIT)"
    while IFS= read -r f; do
        [ -z "$f" ] && continue
        flat="$(printf '%s' "$f" | tr '/' '_')"
        [ -f "$wt/$f" ] && cp "$wt/$f" "$_L/$flat" 2>/dev/null || true
        git -C "$wt" show "${base}:$f" > "$_L/base_$flat" 2>/dev/null || true
        [ -s "$_L/base_$flat" ] || rm -f "$_L/base_$flat"
    done <<EOF
$(sg CHANGED_FILES)
EOF
    echo "artifacts copied; runs.jsonl updated"
}
