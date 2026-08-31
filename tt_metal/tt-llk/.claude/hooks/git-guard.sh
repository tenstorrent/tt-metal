#!/usr/bin/env bash
# git-guard.sh — PreToolUse hook for codegen runs.
#
# Two jobs:
#   1. Log every bash command with an OK/BLOCK verdict, so a run can be audited with
#      `grep BLOCK`.
#   2. During a blind run, deny reads of git history, so a regeneration cannot recover the
#      implementation that was hidden for it.
#
# A codegen worktree shares the parent repo's object store, so every deleted blob is still
# reachable — telling the agent not to look is not enforcement. This covers the shell path;
# permissions.deny in ../settings.json covers Read/Grep/Glob, which never reach a hook and so
# are absent from this log.
#
# Blocking is opt-in: without CODEGEN_BLIND_RUN the hook only logs, so it does not contradict
# the read-only-git policy in ../CLAUDE.md, which the codegen router relies on.
#
# Log line, tab separated — every Bash command, plus Write/Edit only when blocked:
#     <UTC timestamp>  <session id>  <Bash|Write|Edit>  <OK|BLOCK>  <text, newlines flattened>
# Grep the verdict as "\tBLOCK\t", not by column index; the columns have changed once already
# and a hardcoded index fails silently by reporting zero blocks.
#
# Log path: $CODEGEN_GUARD_LOG, else $TMPDIR/codegen-git-guard-<session id>.log, which the
# pipeline copies into the run's LOG_DIR at the end of the run. Setting the variable only
# works for a hand-started session — the hook's environment is fixed when Claude starts.

set -uo pipefail

BLIND="${CODEGEN_BLIND_RUN:-0}"

payload=$(cat)

# Line 1 = session id, 2 = tool name, 3 = file path (Write/Edit), 4.. = the text to scan:
# the command for Bash, the written content for Write/Edit.
meta=$(printf '%s' "$payload" | python3 -c 'import json,sys
try:
    d = json.load(sys.stdin)
    ti = d.get("tool_input") or {}
    tool = str(d.get("tool_name") or "")
    print(str(d.get("session_id") or "nosession"))
    print(tool)
    print(str(ti.get("file_path") or ""))
    if tool == "Bash":
        print(ti.get("command", "") or "")
    elif tool in ("Write", "Edit", "NotebookEdit"):
        print(ti.get("content") or ti.get("new_string") or "")
    else:
        # Read/Grep/Glob: the target path is what matters, not any content.
        keys = ("file_path", "notebook_path", "path", "pattern")
        print(" ".join(str(ti.get(k)) for k in keys if ti.get(k)))
except Exception:
    print("nosession"); print(""); print(""); print("")' 2>/dev/null)
sid=$(printf '%s' "$meta" | sed -n 1p)
tool=$(printf '%s' "$meta" | sed -n 2p)
path=$(printf '%s' "$meta" | sed -n 3p)
cmd=$(printf '%s' "$meta" | tail -n +4)

# One log per run — a shared file would make lines unattributable. Keyed by session id so a
# hand-started session still gets its own file.
LOG="${CODEGEN_GUARD_LOG:-${TMPDIR:-/tmp}/codegen-git-guard-${sid}.log}"

# Blind mode can also be armed mid-session by a marker file, which is how the pipeline does
# it: HIDE_EXISTING_KERNEL is chosen after Claude has started, and a running process's
# environment cannot be changed from outside.
if [ "$BLIND" != "1" ] && [ -f "${TMPDIR:-/tmp}/codegen-blind-run-${sid}" ]; then
    BLIND=1
fi

# A history-reading git subcommand. Tolerates a path prefix (/usr/bin/git), a `command`
# prefix, and leading options such as `-C <dir>` or `--no-pager`.
SUBCMD='(^|[^[:alnum:]_-])git([[:space:]]+(-[^[:space:]]+|-C[[:space:]]+[^[:space:]]+))*[[:space:]]+(log|reflog|rev-list|cat-file|blame|stash|fsck|archive|show|shortlog|whatchanged|for-each-ref)([^[:alnum:]_-]|$)'

# An explicit reference to an older revision, or a raw read of the object store. `worktree`
# itself stays allowed, since the pipeline creates its own. Bare SHAs — which is what stops
# `git worktree add <dir> <old-sha>` — are handled by _has_sha_token below, not here.
REVREF='HEAD~|HEAD\^|@\{|\.git($|[^[:alnum:]_-])'

# Same, minus the bare-SHA clause, for scanning file content: a 7+ hex-digit token is
# common in source (constants, masks) and would false-block legitimate writes.
REVREF_CONTENT='HEAD~|HEAD\^|@\{|\.git($|[^[:alnum:]_-])'

# In script content the subcommand is often not space-separated — `["git", "reflog"]` puts
# quotes and a comma between the tokens. Any run of non-word characters counts as the
# separator, but it cannot span a word, so `git status; cat build.log` still does not match.
SUBCMD_LOOSE='(^|[^[:alnum:]_-])git[^[:alnum:]_]+(log|reflog|rev-list|cat-file|blame|stash|fsck|archive|show|shortlog|whatchanged|for-each-ref)([^[:alnum:]_-]|$)'

# `git branch -v` prints each branch's tip subject — the same disclosure as `git log -1`, and
# how the topk run's analyzer would have learned the op was hidden. Only the verbose forms are
# blocked: setup_worktree.sh needs plain `git branch <name> <base>` and `git branch -D`. A
# branch name can itself contain "-v" (…-quasar-v7), so the flag must be a standalone token.
BRANCHV='(^|[^[:alnum:]_-])git[[:space:]]+branch[^;|&]*[[:space:]](-vv?|--verbose)([[:space:]]|$)'

# Cheap gate: only inspect text that mentions git or a .git path at all.
GITISH='(^|[^[:alnum:]_-])git([^[:alnum:]_-]|$)|\.git($|[^[:alnum:]_-])'

# Naming .git in order to SKIP it is not an attempt to read it. `find . -not -path "./.git/*"`
# and `grep --exclude-dir=.git` are the normal ways to keep a search out of the object store,
# and blocking them produced 2 false positives on the first real topk run. A real history read
# cannot hide inside an exclusion: `git log --exclude=x` still reads as `git log`.
_scrub_exclusions() {
    printf '%s' "$1" \
        | sed -E 's/(-not[[:space:]]+-path|![[:space:]]+-path)[[:space:]]+[^[:space:]]+//g' \
        | sed -E 's/--exclude(-dir)?=[^[:space:]]+//g' \
        | sed -E 's/--exclude(-dir)?[[:space:]]+[^[:space:]]+//g' \
        | sed -E 's/:\(exclude\)[^[:space:]]*//g'
}

# A bare commit SHA as an argument, not as a fragment of something else. Two false-positive
# classes from the first real run drove this shape:
#   * hex inside a path — run ids are 8 hex chars, so every command naming LOG_DIR matched.
#     Hence: whitespace-delimited only.
#   * pure-decimal numbers — every digit is a valid hex digit, so Confluence page ids matched.
#     Hence: the token must contain at least one a-f.
# An all-digit short SHA is therefore missed — accepted, since this is only a backstop behind
# SUBCMD and the HEAD~/.git patterns.
_has_sha_token() {
    printf '%s' "$1" \
        | grep -oE '(^|[[:space:]])[0-9a-f]{7,40}([[:space:]]|:|$)' \
        | tr -d '[:space:]:' \
        | grep -qE '[a-f]'
}

# Writing a script and then running it would defeat a command-text check: the Bash hook only
# sees `bash x.sh`. Authoring it through bash is already caught (the git text is in the
# command), so the remaining route is Write/Edit, which never touches a shell. Scan written
# content too — but only for scripts, so prose that merely mentions `git log` is not blocked.
scan_text=0
scan_revref="$REVREF"
scan_loose=""
case "$tool" in
    Bash)
        scan_text=1
        ;;
    Write|Edit|NotebookEdit)
        scan_revref="$REVREF_CONTENT"
        scan_loose="$SUBCMD_LOOSE"
        case "$path" in
            *.sh | *.bash | *.zsh | *.py | *.pl) scan_text=1 ;;
        esac
        # A shebang makes it a script whatever the extension.
        printf '%s' "$cmd" | sed -n 1p | grep -q '^#!' && scan_text=1
        ;;
    # Read/Grep/Glob are deliberately NOT matched: `permissions.deny` short-circuits before
    # PreToolUse (measured — a denied Read never reaches this hook), so matching them would add
    # no protection while costing ~38 ms per file read (~76 s over a 2000-read run). The deny
    # rules in ../settings.json refuse those reads; the cost is they are not logged here.
esac

verdict=OK
scan_src="$(_scrub_exclusions "$cmd")"
if [ "$BLIND" = "1" ] && [ "$scan_text" = "1" ] && printf '%s' "$scan_src" | grep -qE "$GITISH"; then
    if printf '%s' "$scan_src" | grep -qE "$SUBCMD" || printf '%s' "$scan_src" | grep -qE "$scan_revref" \
        || { [ "$tool" = "Bash" ] && printf '%s' "$scan_src" | grep -qE "$BRANCHV"; }; then
        verdict=BLOCK
    elif [ -n "$scan_loose" ] && printf '%s' "$scan_src" | grep -qE "$scan_loose"; then
        verdict=BLOCK
    elif [ "$tool" = "Bash" ] && _has_sha_token "$scan_src"; then
        verdict=BLOCK
    fi
fi

# Log every bash command, plus any blocked attempt through another tool; logging every Write
# would bury the git-relevant lines. Newlines and tabs are flattened to one greppable line.
if [ "$tool" = "Bash" ] || [ "$verdict" = "BLOCK" ]; then
    entry=$(printf '%s' "$cmd" | tr '\n\t' '  ')
    [ "$tool" != "Bash" ] && entry="$path :: $entry"
    mkdir -p "$(dirname "$LOG")" 2>/dev/null || true
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$sid" "${tool:-?}" "$verdict" "$entry" >> "$LOG" 2>/dev/null || true
fi

if [ "$verdict" = "BLOCK" ]; then
    reason="Blocked by git-guard: this codegen run is a blind regeneration, so reading git history — directly or from a script — is not permitted. git status/add/commit and a bare 'git diff' are allowed."
    printf '%s\n' "$reason" >&2
    python3 -c 'import json,sys; print(json.dumps({"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":sys.argv[1]}}))' "$reason"
fi

exit 0
