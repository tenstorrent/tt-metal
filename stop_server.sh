#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# Unified stop script for the standalone media inference server (server.py).
#
# Supersedes kill_sdxl_server.sh, which only matched the legacy sdxl_server.py /
# sdxl_worker.py process names and therefore never found the unified server.
#
# Shutdown strategy (preserves device cleanup):
#   1. Find the running server.py process tree (main uvicorn process + its
#      multiprocessing worker children — they share the `server.py` cmdline).
#   2. Send SIGTERM to the ROOT process only. uvicorn turns this into FastAPI's
#      lifespan shutdown, which puts a None sentinel on each worker's queue so the
#      workers run `runner.close_device()` and release the mesh/fabric cleanly
#      (identical to pressing Ctrl+C in the launcher terminal).
#   3. Wait GRACEFUL_TIMEOUT for the whole tree to exit.
#   4. SIGKILL any stragglers (last resort — skips device cleanup; a tt-smi reset
#      may then be required).
#
# Usage:
#   ./stop_server.sh                 # graceful stop (recommended)
#   ./stop_server.sh --force         # skip the graceful phase, SIGKILL immediately
#   ./stop_server.sh --timeout 30    # override graceful wait (default 15s)

set -o pipefail

# ---------------------------------------------------------------------------
# Configuration / args
# ---------------------------------------------------------------------------
GRACEFUL_TIMEOUT=15   # seconds to wait for graceful shutdown (device close can be slow)
FORCE=false

while [ $# -gt 0 ]; do
    case "$1" in
        --force)
            FORCE=true
            shift
            ;;
        --timeout)
            GRACEFUL_TIMEOUT="$2"
            shift 2
            ;;
        -h|--help)
            grep '^#' "$0" | grep -vE '^#!|SPDX' | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown argument: $1 (try --help)"
            exit 2
            ;;
    esac
done

# Match the unified server.py invocation, but NOT the legacy sdxl_server.py:
# the char immediately before "server.py" must be a space or slash, so
# ".../sdxl_server.py" (preceded by '_') is excluded.
SERVER_PATTERN='python.*[ /]server\.py'

SCRIPT_NAME="$(basename "$0")"
LOG_PREFIX="[${SCRIPT_NAME}]"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}${LOG_PREFIX} INFO:${NC} $1"; }
log_warn() { echo -e "${YELLOW}${LOG_PREFIX} WARN:${NC} $1"; }
log_error() { echo -e "${RED}${LOG_PREFIX} ERROR:${NC} $1"; }

# ---------------------------------------------------------------------------
# Process discovery
# ---------------------------------------------------------------------------

# All server.py PIDs (main + worker children), excluding this script's own PID.
find_server_pids() {
    pgrep -f "$SERVER_PATTERN" 2>/dev/null | grep -v "^$$\$" | sort -u
}

# The root server process: the one whose parent is NOT itself a server.py
# process (i.e. its parent is the launcher/shell). That's the uvicorn main
# process whose graceful shutdown cascades to the workers.
find_root_pid() {
    local all_pids="$1"
    local pid ppid
    for pid in $all_pids; do
        ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
        if [ -z "$ppid" ]; then
            continue
        fi
        # If the parent is not in our server set, this is a root.
        if ! echo "$all_pids" | grep -q "^${ppid}\$"; then
            echo "$pid"
            return 0
        fi
    done
    return 1
}

get_process_info() {
    ps -p "$1" -o pid,ppid,state,etime,cmd 2>/dev/null | tail -n 1
}

wait_for_all_exit() {
    local pids="$1"
    local timeout="$2"
    local elapsed=0 pid still
    while [ "$elapsed" -lt "$timeout" ]; do
        still=""
        for pid in $pids; do
            if kill -0 "$pid" 2>/dev/null; then
                still="$still $pid"
            fi
        done
        if [ -z "$still" ]; then
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    return 1
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    log_info "Searching for unified media server (server.py) processes..."

    local pids
    pids=$(find_server_pids)

    if [ -z "$pids" ]; then
        log_info "No server.py processes found — nothing to stop."
        exit 0
    fi

    local count
    count=$(echo "$pids" | wc -w)
    log_info "Found ${count} server process(es):"
    local pid
    for pid in $pids; do
        echo "  $(get_process_info "$pid")"
    done
    echo ""

    local used_sigkill=false

    if [ "$FORCE" = "true" ]; then
        log_warn "--force: skipping graceful shutdown, sending SIGKILL to all server processes."
        log_warn "Device cleanup (close_device) is bypassed — a 'tt-smi -r' reset may be required."
        for pid in $pids; do
            kill -KILL "$pid" 2>/dev/null && log_info "Sent SIGKILL to $pid"
        done
        used_sigkill=true
    else
        # Phase 1: graceful SIGTERM to the root process (cascades to workers).
        local root_pid
        root_pid=$(find_root_pid "$pids")

        if [ -n "$root_pid" ]; then
            log_info "Phase 1: graceful shutdown — SIGTERM to root process $root_pid"
            log_info "(uvicorn lifespan will signal workers to close devices cleanly)"
            kill -TERM "$root_pid" 2>/dev/null \
                || log_warn "Failed to SIGTERM $root_pid (may have already exited)"
        else
            log_warn "Could not identify a root process (orphaned workers?); SIGTERM-ing all."
            for pid in $pids; do
                kill -TERM "$pid" 2>/dev/null
            done
        fi

        log_info "Waiting up to ${GRACEFUL_TIMEOUT}s for graceful shutdown..."
        if wait_for_all_exit "$pids" "$GRACEFUL_TIMEOUT"; then
            log_info "All server processes exited gracefully."
            log_info "Shutdown complete."
            exit 0
        fi

        # Phase 2: SIGKILL stragglers.
        local stuck=""
        for pid in $pids; do
            if kill -0 "$pid" 2>/dev/null; then
                stuck="$stuck $pid"
            fi
        done
        log_warn "Processes still alive after ${GRACEFUL_TIMEOUT}s:${stuck}"
        log_info "Phase 2: force killing stragglers (SIGKILL)..."
        for pid in $stuck; do
            local state
            state=$(ps -o state= -p "$pid" 2>/dev/null | tr -d ' ')
            if [ "$state" = "D" ]; then
                log_warn "Process $pid is in uninterruptible sleep (stuck in kernel/driver)."
            fi
            kill -KILL "$pid" 2>/dev/null && log_info "Sent SIGKILL to $pid"
        done
        used_sigkill=true
    fi

    # Final check.
    sleep 2
    local remaining=""
    for pid in $pids; do
        if kill -0 "$pid" 2>/dev/null; then
            remaining="$remaining $pid"
        fi
    done

    if [ -n "$remaining" ]; then
        log_error "Failed to stop:${remaining}"
        log_error "May be stuck in kernel space. Consider 'tt-smi -r' and/or a reboot."
        exit 1
    fi

    if [ "$used_sigkill" = "true" ]; then
        log_warn "Server stopped via SIGKILL — device cleanup was skipped."
        log_warn "If a relaunch fails to open the mesh, reset the chips with: tt-smi -r"
    fi
    log_info "Shutdown complete."
    exit 0
}

trap 'log_warn "Interrupted"; exit 130' INT TERM

main
