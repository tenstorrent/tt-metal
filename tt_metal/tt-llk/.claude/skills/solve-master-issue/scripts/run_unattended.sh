#!/bin/bash
# usage: run_unattended.sh <run-name> <skill arguments...>
# Drive /solve-master-issue from a detached headless session that survives the terminal closing.
#   nohup setsid $SKILL/scripts/run_unattended.sh audit-llk-whbh /path/to/CONFIRMED.json --sync-cmd '…' &
# Each attempt is one `claude -p` turn in auto mode, from $REPO_ROOT (default: cwd). The first attempt
# invokes the skill; every later attempt resumes the SAME session and says "continue from the ledger".
# A turn ends when the model stops — normally because every item is waiting on bots / CI / a device.
# The driver then sleeps $SLEEP seconds (default 900) and resumes, which is how an unattended run
# "wakes up" to handle finished runs: the skill arms nothing that must outlive the turn.
# Stops when $MASTER_ISSUE_DIR/<run-name>/DONE exists (the skill writes it with the final report), or
# after $MAX_ATTEMPTS (default 96 ≈ a day at 15 min). State + logs: $MASTER_ISSUE_DIR/<run-name>/
#   driver.log · session.id · attempt-NN.log (model output) · attempt-NN.err
# Check:  tail -f $MASTER_ISSUE_DIR/<run-name>/driver.log ;  pgrep -af run_unattended
# Stop:   touch $MASTER_ISSUE_DIR/<run-name>/STOP   (the current turn finishes, no new one starts)
set -uo pipefail
name=$1; shift
state=${MASTER_ISSUE_DIR:-$HOME/.claude/master-issues}/$name
mkdir -p "$state"
root=${REPO_ROOT:-$PWD}
sleep_s=${SLEEP:-900}; max=${MAX_ATTEMPTS:-96}
log() { echo "$(date -u +%FT%TZ) $*" >> "$state/driver.log"; }
export MASTER_ISSUE_DIR=${MASTER_ISSUE_DIR:-$HOME/.claude/master-issues}
sid_file=$state/session.id
[ -s "$sid_file" ] || python3 -c 'import uuid;print(uuid.uuid4())' > "$sid_file"
sid=$(cat "$sid_file")
args="$*"
log "start name=$name root=$root session=$sid args=$args"
n=$(ls "$state"/attempt-*.log 2>/dev/null | wc -l)
fresh=1; [ "$n" -gt 0 ] && fresh=0   # a restarted driver resumes the session it had
while [ "$n" -lt "$max" ]; do
    [ -e "$state/DONE" ] && { log "DONE present, exiting"; exit 0; }
    [ -e "$state/STOP" ] && { log "STOP present, exiting"; exit 0; }
    n=$((n + 1)); tag=$(printf 'attempt-%02d' "$n")
    if [ "$fresh" -eq 1 ]; then
        # first turn of a session: invoke the skill; it reconciles the file and ledger itself
        prompt="/solve-master-issue $args --unattended --run-name $name"
        flags=(--session-id "$sid"); fresh=0
    else
        prompt="Resume the /solve-master-issue run '$name' (arguments were: $args --unattended --run-name $name). Re-read the ledger in $state and the findings file marks first, handle everything that finished while you were away (CI runs, bot reviews, builds), reconcile on a fresh start, then continue the scheduler until every item is done, parked, or waiting on something external. Write $state/DONE only when the final report is written and nothing is left to do."
        flags=(--resume "$sid")
    fi
    log "$tag begin"
    ( cd "$root" && claude -p "$prompt" --permission-mode auto "${flags[@]}" \
        --add-dir "$MASTER_ISSUE_DIR" --add-dir "$(dirname "$root")" --add-dir /proj_sw/user_dev/amahmud \
        > "$state/$tag.log" 2> "$state/$tag.err" )
    rc=$?
    log "$tag end rc=$rc"
    if [ "$rc" -ne 0 ] && [ "$n" -gt 1 ] && grep -qiE 'no conversation found|session.*not found' "$state/$tag.err"; then
        log "session $sid lost; next attempt starts a fresh session (ledger-driven resume)"
        python3 -c 'import uuid;print(uuid.uuid4())' > "$sid_file"; sid=$(cat "$sid_file"); fresh=1
    fi
    [ -e "$state/DONE" ] && { log "DONE present, exiting"; exit 0; }
    sleep "$sleep_s"
done
log "max attempts ($max) reached, exiting"
