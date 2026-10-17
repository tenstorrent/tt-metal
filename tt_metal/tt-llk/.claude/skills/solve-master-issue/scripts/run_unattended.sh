#!/bin/bash
# usage: run_unattended.sh <run-name> [--worker <name>] <skill arguments...>
# Drive /solve-master-issue from a detached headless session that survives the terminal closing.
#   nohup setsid $SKILL/scripts/run_unattended.sh audit-llk-whbh /path/to/CONFIRMED.json --sync-cmd '…' &
#   nohup setsid $SKILL/scripts/run_unattended.sh audit-llk-whbh --worker bh /path/to/CONFIRMED.json --arch blackhole … &
# Each attempt is one `claude -p` turn in auto mode, from $REPO_ROOT (default: cwd). The first attempt
# invokes the skill; every later attempt resumes the SAME session and says "continue from the ledger".
# A turn ends when the model stops — normally because every item is waiting on bots / CI / a device.
# The driver then sleeps $SLEEP seconds (default 900) and resumes, which is how an unattended run
# "wakes up" to handle finished runs: the skill arms nothing that must outlive the turn.
#
# Layout under $MASTER_ISSUE_DIR/<run-name>/ (put $MASTER_ISSUE_DIR on storage every box mounts):
#   single worker (no --worker):  driver.log · session.id · attempt-NN.log/.err · ledger.md · DONE · STOP
#   --worker <w>:                 workers/<w>/{driver.log,session.id,attempt-NN.log/.err,ledger.md,DONE,STOP}
#                                 plus the shared claims/ and legs/ dirs the skill uses (findings.py, legs.sh).
#   One driver per box, each with its own --worker name and the same <run-name>; `--worker` is passed
#   through to the skill, which then takes groups by claim instead of by list order.
# Stops when its DONE exists (the skill writes it with the final report), when its own STOP or the run's
# top-level STOP exists (touch $MASTER_ISSUE_DIR/<run-name>/STOP stops every worker after its current
# turn), or after $MAX_ATTEMPTS (default 96 ≈ a day at 15 min).
# Check:  tail -f <state>/driver.log ;  pgrep -af run_unattended
set -uo pipefail
name=$1; shift
worker=
[ "${1:-}" = --worker ] && { worker=$2; shift 2; }
export MASTER_ISSUE_DIR=${MASTER_ISSUE_DIR:-$HOME/.claude/master-issues}
run=$MASTER_ISSUE_DIR/$name
state=$run
[ -n "$worker" ] && state=$run/workers/$worker
mkdir -p "$state"
root=${REPO_ROOT:-$PWD}
sleep_s=${SLEEP:-900}; max=${MAX_ATTEMPTS:-96}
log() { echo "$(date -u +%FT%TZ) $*" >> "$state/driver.log"; }
sid_file=$state/session.id
# fresh = no session to resume. Keyed on session.id, not on old attempt logs: moving a paused run's logs
# under a worker dir must not make the driver try to resume a session that never existed.
fresh=1; [ -s "$sid_file" ] && fresh=0
[ -s "$sid_file" ] || python3 -c 'import uuid;print(uuid.uuid4())' > "$sid_file"
sid=$(cat "$sid_file")
args="$*"
wopt=; [ -n "$worker" ] && wopt=" --worker $worker"
log "start name=$name worker=${worker:-<single>} host=$(hostname) root=$root session=$sid fresh=$fresh args=$args"
n=$(ls "$state"/attempt-*.log 2>/dev/null | wc -l)
stop_requested() { [ -e "$run/STOP" ] || [ -e "$state/STOP" ]; }
while [ "$n" -lt "$max" ]; do
    [ -e "$state/DONE" ] && { log "DONE present, exiting"; exit 0; }
    stop_requested && { log "STOP present, exiting"; exit 0; }
    n=$((n + 1)); tag=$(printf 'attempt-%02d' "$n")
    if [ "$fresh" -eq 1 ]; then
        # first turn of a session: invoke the skill; it reconciles the file and ledger itself
        prompt="/solve-master-issue $args --unattended --run-name $name$wopt"
        flags=(--session-id "$sid"); fresh=0
    else
        prompt="Resume the /solve-master-issue run '$name'${worker:+ as worker '$worker'} (arguments were: $args --unattended --run-name $name$wopt). Re-read the ledger in $state and the findings file marks first, handle everything that finished while you were away (CI runs, bot reviews, builds, remote legs), reconcile on a fresh start, then continue the scheduler until every item is done, parked, or waiting on something external. Write $state/DONE only when the final report is written and nothing is left to do."
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
    stop_requested && { log "STOP present, exiting"; exit 0; }
    sleep "$sleep_s"
done
log "max attempts ($max) reached, exiting"
