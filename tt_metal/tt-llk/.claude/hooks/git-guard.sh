#!/usr/bin/env bash
# git-guard.sh — PreToolUse(Bash) guard for codegen runs.
#
# Two jobs:
#   1. Log every bash command the agent runs, with an OK/BLOCK verdict, so a run can
#      be audited with `grep BLOCK`.
#   2. During a blind run, deny commands that read git history, so a regeneration
#      cannot recover the implementation that was hidden for the run.
#
# Why this exists: a codegen worktree shares the parent repo's object store, so every
# deleted blob is still reachable. Telling the agent not to look is not enforcement.
# This is the enforcement for the shell path; the permissions.deny rules in
# ../settings.json cover the Read/Grep/Glob path, which never touches a shell and so
# never reaches this script (and therefore is NOT recorded in this log).
#
# Blocking is opt-in per run. Without CODEGEN_BLIND_RUN this hook only logs, so it does
# not contradict the read-only-git-is-allowed policy in ../CLAUDE.md and
# ../codegen/CLAUDE.md — the codegen router legitimately uses `git log` / `git show`.
#
# Log line format (tab separated). Every bash command is logged; a Write/Edit is logged only
# when it is blocked, so the git-relevant lines are not buried:
#     <UTC timestamp>  <session id>  <Bash|Write|Edit>  <OK|BLOCK>  <text, newlines flattened>
# Match the verdict as "\tBLOCK\t" rather than by column index — the columns have changed once
# already, and a hardcoded index fails silently by reporting zero blocks.
#
# Log location: $CODEGEN_GUARD_LOG when set, else $TMPDIR/codegen-git-guard-<session id>.log.
# The pipeline does not set that variable — it cannot, since the hook's environment is fixed
# when Claude starts. It instead copies the session-scoped default to "$LOG_DIR/git-guard.log"
# in execute_step_extract_transcripts, so the audit trail ends up stored exactly like
# state.json and the agent transcripts. Set CODEGEN_GUARD_LOG only for a hand-started session.

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
    else:
        print(ti.get("content") or ti.get("new_string") or "")
except Exception:
    print("nosession"); print(""); print(""); print("")' 2>/dev/null)
sid=$(printf '%s' "$meta" | sed -n 1p)
tool=$(printf '%s' "$meta" | sed -n 2p)
path=$(printf '%s' "$meta" | sed -n 3p)
cmd=$(printf '%s' "$meta" | tail -n +4)

# One log per run — a shared file would make lines unattributable. The default is keyed
# by session id so a hand-started session still gets its own file. The pipeline collects
# this file into the run's LOG_DIR at the end of the run (execute_step_extract_transcripts).
LOG="${CODEGEN_GUARD_LOG:-${TMPDIR:-/tmp}/codegen-git-guard-${sid}.log}"

# Blind mode can also be armed by a marker file, which is how the pipeline does it:
# HIDE_EXISTING_KERNEL is chosen *after* Claude has started, and a running process's
# environment cannot be changed from the outside. The marker can be created mid-session.
if [ "$BLIND" != "1" ] && [ -f "${TMPDIR:-/tmp}/codegen-blind-run-${sid}" ]; then
    BLIND=1
fi

# A history-reading git subcommand. Allows for a path prefix (/usr/bin/git), a
# `command` prefix, and leading options such as `-C <dir>` or `--no-pager`.
SUBCMD='(^|[^[:alnum:]_-])git([[:space:]]+(-[^[:space:]]+|-C[[:space:]]+[^[:space:]]+))*[[:space:]]+(log|reflog|rev-list|cat-file|blame|stash|fsck|archive|show)([^[:alnum:]_-]|$)'

# Any explicit reference to an older revision, or a raw read of the object store. The
# bare-SHA pattern is what stops `git worktree add <dir> <old-sha>` and
# `git checkout <old-sha> -- <path>`; `worktree` itself stays allowed because the
# pipeline creates its own worktrees.
REVREF='HEAD~|HEAD\^|@\{|\.git(/|$)|(^|[^[:alnum:]])[0-9a-f]{7,40}([^[:alnum:]]|$)'

# Same, minus the bare-SHA clause, for scanning file content: a 7+ hex-digit token is
# common in source (constants, masks) and would false-block legitimate writes.
REVREF_CONTENT='HEAD~|HEAD\^|@\{|\.git(/|$)'

# In script content the subcommand is often not space-separated —
# `subprocess.run(["git", "reflog"])` puts quotes and a comma between the two tokens. Any
# run of non-word characters counts as the separator. This cannot span an intervening word,
# so `git status; cat build.log` still does not match.
SUBCMD_LOOSE='(^|[^[:alnum:]_-])git[^[:alnum:]_]+(log|reflog|rev-list|cat-file|blame|stash|fsck|archive|show)([^[:alnum:]_-]|$)'

# Cheap gate: only inspect text that mentions git or a .git path at all.
GITISH='(^|[^[:alnum:]_-])git([^[:alnum:]_-]|$)|\.git(/|$)'

# Writing a script and then running it defeats a command-text check: the Bash hook only
# sees `bash x.sh`. Authoring the script *through bash* is already caught (the git text is
# in the command), so the remaining route is the Write/Edit tool, which never touches a
# shell. Scan written content too — but only for scripts, so prose that merely mentions
# `git log` (analysis notes, reports) is not blocked.
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
esac

verdict=OK
if [ "$BLIND" = "1" ] && [ "$scan_text" = "1" ] && printf '%s' "$cmd" | grep -qE "$GITISH"; then
    if printf '%s' "$cmd" | grep -qE "$SUBCMD" || printf '%s' "$cmd" | grep -qE "$scan_revref"; then
        verdict=BLOCK
    elif [ -n "$scan_loose" ] && printf '%s' "$cmd" | grep -qE "$scan_loose"; then
        verdict=BLOCK
    fi
fi

# Log every bash command, plus any blocked attempt through another tool. Logging every
# Write as well would bury the git-relevant lines. Newlines and tabs are flattened so each
# entry is exactly one greppable line.
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
