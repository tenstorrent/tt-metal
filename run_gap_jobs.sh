#!/usr/bin/env bash
#
# Sequentially run the outstanding TTTv2 Milestone A gap jobs in unattended
# Claude Code CLI sessions.
#
#   gap1-finish   finish the interrupted Sampling2D stochastic job   [DONE 2026-08-25]
#   gap2          Prefetcher2D / Galaxy CCL hardware qualification   [cut short by a 429 spend limit]
#   gap2-finish   resume gap2 from its completion handoff            <- current default
#   gap3          batched-prefill policy device evidence (N150/T3K, not the Galaxy)
#
# Jobs are strictly sequential and the Galaxy is confirmed free between them:
# every Galaxy job needs exclusive use of the mesh, so they must never overlap.
# gap3 runs on a different host and is not part of the default job list.
#
# Launch under screen:
#
#     screen -dmS ttgaps /proj_sw/user_dev/ctr-apbernal/tt-metal/run_gap_jobs.sh
#     screen -r ttgaps          # attach
#     # detach with ctrl-a d
#
# Or foreground, to watch it:
#
#     ./run_gap_jobs.sh
#
# Options:
#     --jobs a,b       run these jobs, in this order (default: gap2-finish)
#     --dry-run        run every preflight check, print the commands, execute nothing
#
# The 2026-08-25 run died on an org monthly spend limit (HTTP 429) two thirds of
# the way through gap2 - the auth check below reports that as an auth failure,
# which is what stopped it from burning the remaining jobs.
#
# The previous attempt at job 1 died on an expired Claude credential, so this
# script refuses to start a job without a live auth check, and re-checks between
# jobs. A credential that dies mid-job is detected and reported rather than
# silently burning the next job.

set -uo pipefail

REPO="/proj_sw/user_dev/ctr-apbernal/tt-metal"
BRIEFS="$REPO/tttv2_milestone_a_gap_briefs"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$REPO/tttv2_gap_jobs_runs/$STAMP"
DRIVER_LOG="$RUN_DIR/driver.log"

MODEL="opus"
JOB_TIMEOUT_GAP1="${JOB_TIMEOUT_GAP1:-7200}"    # 2 h  - write-up work, no long device runs
JOB_TIMEOUT_GAP2="${JOB_TIMEOUT_GAP2:-43200}"   # 12 h - device job with resets and repeats
JOB_TIMEOUT_GAP3="${JOB_TIMEOUT_GAP3:-21600}"   # 6 h  - non-Galaxy host
AUTH_TIMEOUT=120

# gap1-finish and gap2 completed on 2026-08-25; gap2 was cut short by a spend
# limit, so the default now resumes it from its completion handoff.
JOBS="gap2-finish"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --jobs)    JOBS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "$RUN_DIR"

log() { printf '%s  %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "$DRIVER_LOG"; }
fail() { log "FATAL: $*"; log "driver log: $DRIVER_LOG"; exit 1; }

# ---------------------------------------------------------------- preflight --

check_tools() {
    command -v claude >/dev/null || fail "claude CLI not on PATH"
    command -v tt-smi >/dev/null || log "WARNING: tt-smi not on PATH; device checks degraded"
    [[ -d "$REPO/.git" ]] || fail "$REPO is not a git repo"
    [[ -d "$BRIEFS" ]] || fail "briefs directory missing: $BRIEFS"
}

# The failure mode that killed the last run. Cheap, and worth doing every time.
check_auth() {
    local out
    out="$(timeout "$AUTH_TIMEOUT" claude -p 'Reply with exactly: AUTH_OK' \
             --model "$MODEL" 2>&1)"
    if [[ "$out" == *AUTH_OK* ]]; then
        log "auth OK"
        return 0
    fi
    log "auth check FAILED. Response was:"
    printf '%s\n' "$out" | sed 's/^/    /' | tee -a "$DRIVER_LOG"
    return 1
}

device_count() { ls /dev/tenstorrent 2>/dev/null | wc -l; }

check_device_free() {
    local n holders
    n="$(device_count)"
    [[ "$n" == "32" ]] || fail "expected 32 /dev/tenstorrent nodes, found $n - not a complete (8,4) Galaxy"
    # Exclude our own tree. The claude wrapper carries the job prompt on its command
    # line, so a bare 'pgrep -f pytest' matches it and it reads as a stray test
    # process - which is how the 2026-08-26 run talked itself into self-termination.
    holders="$(pgrep -af 'pytest|ttnn' \
        | grep -v -e grep -e run_gap_jobs -e 'claude -p' -e '^[0-9]* timeout ' || true)"
    if [[ -n "$holders" ]]; then
        log "device is held by:"
        printf '%s\n' "$holders" | sed 's/^/    /' | tee -a "$DRIVER_LOG"
        return 1
    fi
    log "device free: 32 nodes, no pytest/ttnn processes"
    return 0
}

wait_device_free() {
    local waited=0 limit=600
    while ! check_device_free; do
        (( waited >= limit )) && return 1
        log "waiting for the device to free up (${waited}s/${limit}s)..."
        sleep 30
        waited=$(( waited + 30 ))
    done
    return 0
}

reset_galaxy() {
    local tag="$1"
    log "running tt-smi -glx_reset ($tag)"
    timeout 600 tt-smi -glx_reset > "$RUN_DIR/reset_${tag}.log" 2>&1
    local rc=$?
    if grep -q "Re-initialized 32 boards" "$RUN_DIR/reset_${tag}.log"; then
        log "reset OK (32 boards re-initialized)"
        return 0
    fi
    log "reset did NOT confirm 32 boards (exit=$rc); see $RUN_DIR/reset_${tag}.log"
    return 1
}

# ---------------------------------------------------------------- job runner --

# Each job gets the brief by path and is told to execute it. The briefs are
# self-contained and carry their own prohibitions, run procedure and finish
# condition, so the prompt stays thin on purpose.
job_prompt() {
    local brief="$1"
    cat <<EOF
Read $brief in full and carry out the work it specifies, end to end.

That document is your complete instruction set: it defines the scope, the run
procedure, the prohibitions and the finish condition. Follow it literally. It was
written for exactly this task; do not substitute your own plan for it.

You are running unattended on a shared Wormhole Galaxy host. Nobody will answer
questions, so make the judgment calls the brief already made for you and do not
stop to ask. If you hit something the brief genuinely does not cover, choose the
most conservative option, record the decision, and continue.

Two things override everything else:
  - exactly one test process may touch the device at any moment;
  - never fabricate a result. An honest BLOCKED with logs beats an invented pass.

DO NOT KILL YOUR OWN PROCESS TREE. You are running inside
"timeout ... claude -p <this prompt>", launched by run_gap_jobs.sh. That wrapper's
command line contains this entire prompt, so it matches loose patterns such as
'pgrep -af pytest' and will appear in your process listings looking like a stray
test process. A previous run read exactly that line, took it for a stuck test, ran
'kill -TERM' on it and killed itself 16 minutes in.

Before you signal any PID:
  - your own tree is PID $$ (the driver) and its "timeout"/"claude" children;
  - the environment variable GAP_JOB_DRIVER_PID also holds the driver PID;
  - confirm the target is really a test process with
        ps -o pid=,ppid=,comm=,args= -p <pid>
    and require comm to be python/python3/pytest. Never signal a PID whose comm is
    "claude", "timeout", "bash" or "screen";
  - prefer targeting the exact file, e.g.
        pkill -f 'python.*pytest.*<the test file you launched>'
    and confirm with pgrep first that it matches only what you intend.

When you are done, print the absolute path of the REPORT.md you wrote as your
final line.
EOF
}

run_job() {
    local name="$1" brief="$2" timeout_s="$3"
    local log_file="$RUN_DIR/${name}.stream.jsonl"
    local txt_file="$RUN_DIR/${name}.txt"

    [[ -f "$brief" ]] || fail "brief not found: $brief"

    log "==================== job: $name"
    log "brief:   $brief"
    log "timeout: ${timeout_s}s"
    log "stream:  $log_file"

    if (( DRY_RUN )); then
        log "[dry-run] would run: claude -p <prompt> --model $MODEL --dangerously-skip-permissions"
        return 0
    fi

    local started
    started="$(date -u +%s)"

    # cd into the repo so relative paths in the brief resolve, and so the session
    # picks up project settings and CLAUDE.md. GAP_JOB_DRIVER_PID lets the agent
    # recognise its own tree instead of mistaking it for a stray test process.
    # The prompt goes in on stdin, never as an argv element. Passing it as an
    # argument put its whole text on the wrapper's command line, so 'pgrep -af
    # pytest' matched the claude process itself - the 2026-08-26 run read that
    # line, mistook it for a stuck test and killed itself with it.
    (
        cd "$REPO" || exit 1
        export GAP_JOB_DRIVER_PID=$$
        job_prompt "$brief" \
          | timeout --signal=TERM --kill-after=300 "$timeout_s" \
                claude -p \
                    --model "$MODEL" \
                    --dangerously-skip-permissions \
                    --output-format stream-json \
                    --include-partial-messages \
                    --verbose
    ) > "$log_file" 2>&1
    local rc=$?

    local elapsed=$(( $(date -u +%s) - started ))
    log "job $name exited rc=$rc after ${elapsed}s"

    # Pull the human-readable trail out of the stream for quick reading.
    if command -v python3 >/dev/null; then
        python3 - "$log_file" > "$txt_file" 2>/dev/null <<'PY'
import json, sys
for line in open(sys.argv[1], errors="replace"):
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        event = json.loads(line)
    except ValueError:
        continue
    if event.get("type") == "assistant":
        for block in event.get("message", {}).get("content", []):
            if block.get("type") == "text" and block.get("text", "").strip():
                print(block["text"])
    elif event.get("type") == "result":
        print("\n--- RESULT ---")
        print(event.get("result", ""))
PY
        log "readable transcript: $txt_file"
    fi

    # A session id makes the run resumable by hand: claude --resume <id>
    local session
    session="$(grep -o '"session_id":"[^"]*"' "$log_file" 2>/dev/null | head -1 | cut -d'"' -f4)"
    [[ -n "$session" ]] && log "session id: $session   (resume with: claude --resume $session)"

    if (( rc == 124 || rc == 137 )); then
        log "job $name hit its ${timeout_s}s timeout"
    fi
    return $rc
}

# --------------------------------------------------------------------- main --

log "run directory: $RUN_DIR"
log "repo:          $REPO"
log "jobs:          $JOBS"
log "model:         $MODEL"

check_tools

git -C "$REPO" rev-parse HEAD    | sed 's/^/HEAD:   /' | tee -a "$DRIVER_LOG"
git -C "$REPO" branch --show-current | sed 's/^/branch: /' | tee -a "$DRIVER_LOG"
git -C "$REPO" status --short > "$RUN_DIR/git_status_before.txt"
log "working tree snapshot: $RUN_DIR/git_status_before.txt"
log "NOTE: the tree is expected to be dirty - the uncommitted fixes are the deliverable."

check_auth || fail "not authenticated. Run 'claude' interactively and sign in, then re-launch."
check_device_free || fail "device is not free; clear it before launching."

if command -v tt-smi >/dev/null; then
    tt-smi -ls > "$RUN_DIR/00_tt_smi_before.log" 2>&1
    log "tt-smi baseline: $RUN_DIR/00_tt_smi_before.log"
fi

overall_rc=0
IFS=',' read -ra JOB_LIST <<< "$JOBS"

for job in "${JOB_LIST[@]}"; do
    case "$job" in
        gap1-finish) brief="$BRIEFS/gap1_completion_handoff.md";               tmo="$JOB_TIMEOUT_GAP1" ;;
        gap2)        brief="$BRIEFS/gap2_prefetcher2d_galaxy_ccl_hardware.md"; tmo="$JOB_TIMEOUT_GAP2" ;;
        gap2-finish) brief="$BRIEFS/gap2_completion_handoff.md";               tmo="$JOB_TIMEOUT_GAP2" ;;
        gap3)        brief="$BRIEFS/gap3_batched_prefill_physical32_trace.md"; tmo="$JOB_TIMEOUT_GAP3" ;;
        *) log "skipping unknown job: $job"; continue ;;
    esac

    # Re-check auth before every job: the credential that expired last time did
    # so mid-run, and starting a 12 h device job on a dead token wastes the slot.
    if ! check_auth; then
        log "auth is dead before job '$job'. Stopping rather than burning the job."
        log "Sign in again ('claude' interactively), then re-launch with: $0 --jobs $(IFS=,; echo "${JOB_LIST[*]}")"
        overall_rc=1
        break
    fi

    if ! wait_device_free; then
        log "device still held before job '$job'; attempting a reset"
        reset_galaxy "before_$job" || { log "reset failed; stopping"; overall_rc=1; break; }
        wait_device_free || { log "device still held after reset; stopping"; overall_rc=1; break; }
    fi

    run_job "$job" "$brief" "$tmo"
    rc=$?
    (( rc != 0 )) && overall_rc=$rc

    # Whatever happened, do not hand the next job a device that is still held.
    if ! wait_device_free; then
        log "device held after job '$job'; killing stragglers"
        pkill -TERM -f pytest; sleep 20; pkill -KILL -f pytest 2>/dev/null
        wait_device_free || reset_galaxy "after_$job" || log "WARNING: could not restore a clean device"
    fi

    if (( rc != 0 )); then
        log "job '$job' did not exit cleanly (rc=$rc)."
        if ! check_auth; then
            log "auth is dead - that is the likely cause. Remaining jobs are NOT started."
            break
        fi
        log "auth is fine, so this was a task-level failure. Continuing to the next job."
    fi
done

git -C "$REPO" status --short > "$RUN_DIR/git_status_after.txt"
if command -v tt-smi >/dev/null; then
    tt-smi -ls > "$RUN_DIR/99_tt_smi_after.log" 2>&1
fi

log "==================== all jobs finished (overall rc=$overall_rc)"
log "run directory: $RUN_DIR"
log "changed files: diff $RUN_DIR/git_status_before.txt $RUN_DIR/git_status_after.txt"
for f in "$RUN_DIR"/*.txt "$RUN_DIR"/*.log; do
    [[ -e "$f" ]] && log "  $f"
done

exit "$overall_rc"
