#!/bin/bash
# tt_device_pool.sh - per-card device pool (sourced by tt_safe_run.sh for
# run_safe_pytest.sh, tt-probe.sh and eval/eval_test_runner.sh)
#
# Source this file; do not execute it. It turns the old single global lock
# (/tmp/tt-device.lock + tt-smi -r of the whole box) into one lock per card so
# independent jobs can run on independent cards at the same time.
#
# Selector (what the caller passes to ttpool_acquire):
#   auto   - DEFAULT. Lowest free card wins: try `flock -n` on card 0, then 1, 2, ...
#            If every card is busy, re-scan once a second, still lowest-first.
#            No round-robin and no queue fairness among waiters, by design.
#   <N>    - Wait for that specific card (UMD logical id, see below).
#   mesh   - Take EVERY card. Locks are taken in ascending order and held as they
#            are obtained, so a waiting mesh job steadily drains the pool: new
#            `auto` jobs skip the cards it already holds. Ascending order on both
#            sides makes deadlock impossible.
#
# Card ids are UMD logical ids: the "UMD Chip ID" column of `tt-smi -ls`, the
# same numbering `tt-smi -r <id>` and TT_VISIBLE_DEVICES use. They are NOT the
# /dev/tenstorrent/<n> minor numbers (on a 4x p150a box 0->/dev/1, 3->/dev/0).
# The card count is taken from /dev/tenstorrent/*.
#
# Per-card state and isolation:
#   /tmp/tt-device-<id>.lock   flock, held on a dedicated fd for the whole run
#   /tmp/tt-device-<id>.dirty  "may need a reset"; whoever takes the card next resets it
#   TT_VISIBLE_DEVICES=<id>    process, UMD AND tt-exalens/tt-triage see only this card,
#                              as device 0. (TT_METAL_VISIBLE_DEVICES is NOT consumed for
#                              device selection - do not use it.)
#   TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS=127.0.0.1:<50051+id>
#                              every process otherwise fights over :50051 and tt-triage
#                              then talks to the wrong process and fails.
#   TT_METAL_LOGS_PATH=<repo>/generated/dev<id>
#                              watcher / inspector / fabric logs per card.
#   tt-smi -r <id>             resets one card without disturbing the others.
#   TT_METAL_PROFILER_DIR=<repo>/generated/dev<id>/profiler and an explicit Tracy
#                              capture port <8086+id>: the default profiler dir is
#                              shared (concurrent runs corrupt each other's CSVs) and
#                              tracy's get_available_port() races (all pick 8086).
#   The kernel cache (TT_METAL_CACHE) stays SHARED: jit_build publishes every
#   artifact with an atomic rename, so concurrent cold compiles are safe.
#
# Mesh runs use all cards visible, label "mesh", Inspector port 50100, Tracy port
# 8100, logs under generated/mesh, and reset every card on a hang.
#
# Simulator mode is the caller's business: do not call ttpool_acquire on sim.
#
# API (all functions print nothing on success unless noted; TTPOOL_LOG_PREFIX
# prefixes every line they print, default "TTPOOL"):
#   ttpool_acquire <selector>       blocks until the card(s) are held. Sets
#                                   TTPOOL_CARDS, TTPOOL_LABEL, TTPOOL_IS_MESH.
#   ttpool_setup_env <repo_dir>     exports the per-card env, sets TTPOOL_LOGS_DIR,
#                                   TTPOOL_TRIAGE_OUT_DIR, TTPOOL_TRIAGE_REPORT,
#                                   TTPOOL_TRIAGE_EXTRA_ARGS, TTPOOL_WATCHER_LOG,
#                                   TTPOOL_PROFILER_DIR, TTPOOL_TRACY_PORT.
#   ttpool_reset_if_dirty           reset held card(s) if any dirty flag is present.
#                                   Returns 1 if the reset failed.
#   ttpool_mark_dirty / ttpool_clear_dirty
#   ttpool_reset                    tt-smi -r on the held card(s). Returns tt-smi's rc.
#   ttpool_publish_triage_report    ln -sfn the per-card report to the legacy path
#                                   generated/tt-triage/triage.txt (latest hang wins).
#   ttpool_holder_info <id>         "pid=... cmd=..." of the process holding a card's
#                                   lock, or empty.
#   ttpool_release                  close the lock fds (normally implicit at process exit).
#
# TTPOOL_ACQUIRE_TIMEOUT=<seconds> (env, default 0 = wait forever) bounds ttpool_acquire;
# on expiry it returns 2 with an error line, holding nothing.

TTPOOL_LOG_PREFIX="${TTPOOL_LOG_PREFIX:-TTPOOL}"
TTPOOL_CARDS=""
TTPOOL_LABEL=""
TTPOOL_IS_MESH=false
TTPOOL_LOCK_FDS=""
TTPOOL_MESH_RPC_PORT=50100
TTPOOL_MESH_TRACY_PORT=8100
TTPOOL_SINGLE_RPC_BASE=50051
TTPOOL_SINGLE_TRACY_BASE=8086
TTPOOL_ACQUIRE_TIMEOUT="${TTPOOL_ACQUIRE_TIMEOUT:-0}"
_TTPOOL_DEADLINE=0

_ttpool_say() { echo "${TTPOOL_LOG_PREFIX}: $*"; }

ttpool_card_count() {
    ls /dev/tenstorrent/ 2>/dev/null | grep -c '^[0-9][0-9]*$'
}

ttpool_lock_file() { echo "/tmp/tt-device-$1.lock"; }
ttpool_dirty_flag() { echo "/tmp/tt-device-$1.dirty"; }

# Find the PID holding an flock on a lock path. Tries lslocks first (fast, global
# namespace); falls back to scanning /proc/*/fd for the file + an active FLOCK in
# fdinfo (works inside PID namespaces where lslocks reports pid 0).
_ttpool_find_lock_holder() {
    local lock_path="$1" pid
    pid=$(lslocks --noheadings --raw --output PID,PATH 2>/dev/null \
        | awk -v p="$lock_path" '$2==p && $1!="0" {print $1; exit}')
    if [[ -n "$pid" ]]; then echo "$pid"; return 0; fi
    local pid_dir fd_link fd_num target
    for pid_dir in /proc/[0-9]*; do
        for fd_link in "$pid_dir"/fd/*; do
            [ -L "$fd_link" ] || continue
            target=$(readlink "$fd_link" 2>/dev/null) || continue
            [ "$target" = "$lock_path" ] || continue
            fd_num=${fd_link##*/}
            if grep -q '^lock:.*FLOCK' "$pid_dir/fdinfo/$fd_num" 2>/dev/null; then
                echo "${pid_dir##*/}"
                return 0
            fi
        done
    done
    return 1
}

ttpool_holder_info() {
    local pid cmd ppid pcmd
    pid=$(_ttpool_find_lock_holder "$(ttpool_lock_file "$1")") || return 0
    [[ -n "$pid" && -d /proc/$pid ]] || return 0
    cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | cut -c1-200)
    ppid=$(awk '{print $4}' /proc/$pid/stat 2>/dev/null)
    if [[ -n "$ppid" && "$ppid" -gt 1 && -d /proc/$ppid ]]; then
        pcmd=$(tr '\0' ' ' < /proc/$ppid/cmdline 2>/dev/null | cut -c1-150)
        echo "holder pid=$pid cmd=\"$cmd\" parent pid=$ppid cmd=\"$pcmd\""
    else
        echo "holder pid=$pid cmd=\"$cmd\""
    fi
}

# Open a new fd on the lock file and record it. Sets _TTPOOL_FD.
# Lock and dirty files live in /tmp and are shared by every user of the box, so they
# are made world-writable on creation (best effort; a file another user created with
# a stricter mode is reported instead of silently blocking that card for us).
_ttpool_open_fd() {
    local fd path
    path="$(ttpool_lock_file "$1")"
    [[ -e "$path" ]] || { : >"$path" 2>/dev/null && chmod a+rw "$path" 2>/dev/null; }
    if ! exec {fd}>"$path"; then
        _ttpool_say "ERROR: cannot open ${path} for writing (owned by $(stat -c %U "$path" 2>/dev/null || echo '?')); ask them to chmod a+rw it"
        return 1
    fi
    _TTPOOL_FD=$fd
}

# True once the optional acquire deadline has passed.
_ttpool_timed_out() { [[ "$_TTPOOL_DEADLINE" -gt 0 && $(date +%s) -ge "$_TTPOOL_DEADLINE" ]]; }

# Blocking wait for one card. Waits in slices of at most 20 s so the holder report
# stays fresh, and never past the acquire deadline. Returns 2 on deadline expiry.
_ttpool_wait_card() {
    local card="$1" fd waited=0 slice info
    _ttpool_open_fd "$card" || return 1
    fd=$_TTPOOL_FD
    while :; do
        slice=20
        if [[ "$_TTPOOL_DEADLINE" -gt 0 ]]; then
            slice=$(( _TTPOOL_DEADLINE - $(date +%s) ))
            [[ $slice -gt 20 ]] && slice=20
        fi
        if [[ $slice -gt 0 ]] && flock -w "$slice" "$fd"; then
            break
        fi
        if _ttpool_timed_out; then
            exec {fd}>&-
            _ttpool_say "ERROR: gave up waiting for card ${card} after ${TTPOOL_ACQUIRE_TIMEOUT}s"
            return 2
        fi
        waited=$((waited + slice))
        info=$(ttpool_holder_info "$card")
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${TTPOOL_LOG_PREFIX}: waiting for card ${card} (${waited}s) — ${info:-holder unknown}"
    done
    TTPOOL_LOCK_FDS="${TTPOOL_LOCK_FDS} ${fd}"
}

ttpool_release() {
    local fd
    for fd in $TTPOOL_LOCK_FDS; do exec {fd}>&-; done
    TTPOOL_LOCK_FDS=""
}

ttpool_acquire() {
    local selector="${1:-auto}" num_cards c fd t0 announced=false last_report=0 now
    num_cards=$(ttpool_card_count)
    if [[ "$num_cards" -eq 0 ]]; then
        _ttpool_say "ERROR: no /dev/tenstorrent/* devices found"
        return 1
    fi
    _TTPOOL_DEADLINE=0
    [[ "$TTPOOL_ACQUIRE_TIMEOUT" -gt 0 ]] && _TTPOOL_DEADLINE=$(( $(date +%s) + TTPOOL_ACQUIRE_TIMEOUT ))
    case "$selector" in
        auto)
            t0=$(date +%s)
            _ttpool_say "acquiring a card (pool of ${num_cards}, lowest free wins)..."
            while :; do
                for ((c = 0; c < num_cards; c++)); do
                    _ttpool_open_fd "$c" || return 1
                    fd=$_TTPOOL_FD
                    if flock -n "$fd"; then
                        TTPOOL_CARDS="$c"; TTPOOL_LOCK_FDS=" $fd"
                        break 2
                    fi
                    exec {fd}>&-
                done
                if _ttpool_timed_out; then
                    _ttpool_say "ERROR: no card became free within ${TTPOOL_ACQUIRE_TIMEOUT}s"
                    return 2
                fi
                now=$(date +%s)
                if [[ "$announced" == false || $((now - last_report)) -ge 60 ]]; then
                    announced=true; last_report=$now
                    _ttpool_say "all ${num_cards} cards busy ($((now - t0))s), waiting for the lowest one to free up..."
                    for ((c = 0; c < num_cards; c++)); do
                        _ttpool_say "  card ${c}: $(ttpool_holder_info "$c")"
                    done
                fi
                sleep 1
            done
            TTPOOL_LABEL="dev${TTPOOL_CARDS}"
            _ttpool_say "Device lock acquired: card ${TTPOOL_CARDS} (auto, waited $(( $(date +%s) - t0 ))s)"
            ;;
        mesh)
            TTPOOL_IS_MESH=true
            TTPOOL_LABEL="mesh"
            _ttpool_say "acquiring ALL ${num_cards} cards for a mesh run (ascending, hold-and-wait)..."
            for ((c = 0; c < num_cards; c++)); do
                if ! _ttpool_wait_card "$c"; then ttpool_release; TTPOOL_CARDS=""; return 2; fi
                TTPOOL_CARDS="${TTPOOL_CARDS:+$TTPOOL_CARDS }$c"
                _ttpool_say "  holding card ${c}"
            done
            _ttpool_say "Device lock acquired: cards ${TTPOOL_CARDS} (mesh)"
            ;;
        *)
            if ! [[ "$selector" =~ ^[0-9]+$ ]]; then
                _ttpool_say "ERROR: bad device selector '${selector}' (want auto, mesh or a UMD id 0..$((num_cards - 1)))"
                return 1
            fi
            selector=$((10#$selector))  # canonical decimal: "00" == "0", one lock per card
            if [[ "$selector" -ge "$num_cards" ]]; then
                _ttpool_say "ERROR: card ${selector} does not exist (found ${num_cards} cards: 0..$((num_cards - 1)))"
                return 1
            fi
            _ttpool_say "waiting for card ${selector}..."
            _ttpool_wait_card "$selector" || return 2
            TTPOOL_CARDS="$selector"
            TTPOOL_LABEL="dev${selector}"
            _ttpool_say "Device lock acquired: card ${selector}"
            ;;
    esac
    return 0
}

ttpool_setup_env() {
    local repo_dir="$1" rpc_port
    [[ -n "$TTPOOL_LABEL" ]] || { _ttpool_say "ERROR: ttpool_setup_env before ttpool_acquire"; return 1; }
    TTPOOL_LOGS_DIR="${repo_dir}/generated/${TTPOOL_LABEL}"
    TTPOOL_TRIAGE_OUT_DIR="${TTPOOL_LOGS_DIR}/tt-triage"
    TTPOOL_TRIAGE_REPORT="${TTPOOL_TRIAGE_OUT_DIR}/triage.txt"
    TTPOOL_WATCHER_LOG="${TTPOOL_LOGS_DIR}/generated/watcher/watcher.log"
    TTPOOL_PROFILER_DIR="${TTPOOL_LOGS_DIR}/profiler"
    if [[ "$TTPOOL_IS_MESH" == true ]]; then
        export TT_VISIBLE_DEVICES="${TTPOOL_CARDS// /,}"
        rpc_port=$TTPOOL_MESH_RPC_PORT
        TTPOOL_TRACY_PORT=$TTPOOL_MESH_TRACY_PORT
    else
        export TT_VISIBLE_DEVICES="$TTPOOL_CARDS"
        rpc_port=$((TTPOOL_SINGLE_RPC_BASE + TTPOOL_CARDS))
        TTPOOL_TRACY_PORT=$((TTPOOL_SINGLE_TRACY_BASE + TTPOOL_CARDS))
    fi
    export TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS="127.0.0.1:${rpc_port}"
    export TT_METAL_LOGS_PATH="$TTPOOL_LOGS_DIR"
    # The dispatch-timeout hook inherits TT_VISIBLE_DEVICES, so tt-exalens only opens
    # our card(s); these two args point tt-triage at OUR inspector, not :50051.
    TTPOOL_TRIAGE_EXTRA_ARGS="--inspector-rpc-port=${rpc_port} --inspector-log-path=${TTPOOL_LOGS_DIR}/generated/inspector"
    mkdir -p "$TTPOOL_TRIAGE_OUT_DIR"
}

ttpool_mark_dirty() {
    local c f
    for c in $TTPOOL_CARDS; do
        f="$(ttpool_dirty_flag "$c")"
        [[ -e "$f" ]] || { : >"$f" 2>/dev/null && chmod a+rw "$f" 2>/dev/null; }
    done
}
ttpool_clear_dirty() { local c; for c in $TTPOOL_CARDS; do rm -f "$(ttpool_dirty_flag "$c")"; done; }

ttpool_reset() {
    # shellcheck disable=SC2086
    tt-smi -r $TTPOOL_CARDS
}

ttpool_reset_if_dirty() {
    local c dirty=""
    for c in $TTPOOL_CARDS; do [[ -f "$(ttpool_dirty_flag "$c")" ]] && dirty="${dirty:+$dirty }$c"; done
    [[ -n "$dirty" ]] || return 0
    _ttpool_say "card(s) ${dirty} marked dirty from a previous run, resetting card(s) ${TTPOOL_CARDS}..."
    if ! ttpool_reset; then
        _ttpool_say "ERROR: tt-smi -r ${TTPOOL_CARDS} failed"
        return 1
    fi
    ttpool_clear_dirty
    _ttpool_say "Device reset complete"
}

# Keep the documented legacy location pointing at the most recent hang report.
ttpool_publish_triage_report() {
    local repo_dir="$1" legacy_dir="${1}/generated/tt-triage"
    [[ -f "$TTPOOL_TRIAGE_REPORT" ]] || return 0
    mkdir -p "$legacy_dir" 2>/dev/null
    ln -sfn "$TTPOOL_TRIAGE_REPORT" "${legacy_dir}/triage.txt" 2>/dev/null || true
}
