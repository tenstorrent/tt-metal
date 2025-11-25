#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# Kill script for SDXL standalone server
# Handles graceful and forced shutdown of server and worker processes

set -o pipefail

# Configuration
GRACEFUL_TIMEOUT=5  # Seconds to wait for graceful shutdown
SCRIPT_NAME="$(basename "$0")"
LOG_PREFIX="[${SCRIPT_NAME}]"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}${LOG_PREFIX} INFO:${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}${LOG_PREFIX} WARN:${NC} $1"
}

log_error() {
    echo -e "${RED}${LOG_PREFIX} ERROR:${NC} $1"
}

# Find all SDXL-related processes
find_sdxl_processes() {
    # Find Python processes running SDXL scripts
    # Look for: sdxl_server.py, sdxl_worker.py, uvicorn with sdxl
    local pids=$(pgrep -f "python.*sdxl_(server|worker)" 2>/dev/null)

    # Also find uvicorn processes serving the SDXL app
    local uvicorn_pids=$(pgrep -f "uvicorn.*sdxl" 2>/dev/null)

    # Combine and deduplicate
    echo "$pids $uvicorn_pids" | tr ' ' '\n' | sort -u | grep -v '^$'
}

# Check if process is responding
is_process_responsive() {
    local pid=$1
    if [ ! -d "/proc/$pid" ]; then
        return 1  # Process doesn't exist
    fi

    # Check if process is in uninterruptible sleep (D state) - stuck
    local state=$(ps -o state= -p "$pid" 2>/dev/null | tr -d ' ')
    if [ "$state" = "D" ]; then
        return 1  # Process is stuck
    fi

    return 0  # Process is responsive
}

# Send signal to process
send_signal() {
    local pid=$1
    local sig=$2
    local sig_name=$3

    if kill -$sig "$pid" 2>/dev/null; then
        log_info "Sent $sig_name to process $pid"
        return 0
    else
        log_warn "Failed to send $sig_name to process $pid (may have already exited)"
        return 1
    fi
}

# Wait for process to exit
wait_for_exit() {
    local pid=$1
    local timeout=$2
    local elapsed=0

    while [ $elapsed -lt $timeout ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            return 0  # Process exited
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    return 1  # Timeout
}

# Get process info for display
get_process_info() {
    local pid=$1
    ps -p "$pid" -o pid,ppid,state,etime,cmd 2>/dev/null | tail -n 1
}

# Main execution
main() {
    log_info "Searching for SDXL server processes..."

    # Find all SDXL processes
    local pids=($(find_sdxl_processes))

    if [ ${#pids[@]} -eq 0 ]; then
        log_info "No SDXL server processes found"
        exit 0
    fi

    log_info "Found ${#pids[@]} SDXL process(es):"
    for pid in "${pids[@]}"; do
        echo "  $(get_process_info "$pid")"
    done
    echo ""

    # Phase 1: Try graceful shutdown with SIGTERM
    log_info "Phase 1: Attempting graceful shutdown (SIGTERM)..."
    local alive_pids=()

    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            send_signal "$pid" TERM "SIGTERM"
            alive_pids+=("$pid")
        fi
    done

    if [ ${#alive_pids[@]} -eq 0 ]; then
        log_info "All processes already terminated"
    else
        log_info "Waiting ${GRACEFUL_TIMEOUT} seconds for graceful shutdown..."
        sleep "$GRACEFUL_TIMEOUT"

        # Check which processes are still alive
        local stuck_pids=()
        for pid in "${alive_pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                stuck_pids+=("$pid")
            else
                log_info "Process $pid exited gracefully"
            fi
        done

        # Phase 2: Force kill stuck processes
        if [ ${#stuck_pids[@]} -gt 0 ]; then
            log_warn "${#stuck_pids[@]} process(es) did not respond to SIGTERM"
            log_info "Phase 2: Force killing stuck processes (SIGKILL)..."

            for pid in "${stuck_pids[@]}"; do
                local state=$(ps -o state= -p "$pid" 2>/dev/null | tr -d ' ')
                if [ "$state" = "D" ]; then
                    log_warn "Process $pid is in uninterruptible sleep (stuck in kernel/driver)"
                fi

                send_signal "$pid" KILL "SIGKILL"
            done

            # Wait briefly for SIGKILL to take effect
            sleep 2

            # Final check
            local remaining_pids=()
            for pid in "${stuck_pids[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    remaining_pids+=("$pid")
                    log_error "Process $pid still alive after SIGKILL!"
                else
                    log_info "Process $pid terminated"
                fi
            done

            if [ ${#remaining_pids[@]} -gt 0 ]; then
                log_error "Failed to kill ${#remaining_pids[@]} process(es)"
                log_error "These processes may be stuck in kernel space. Consider system reboot."
                # Don't exit yet - still try to reset devices
            fi
        else
            log_info "All processes exited gracefully"
        fi
    fi

    log_info "Graceful shutdown complete"

    # Exit code based on results
    if [ ${#remaining_pids[@]} -gt 0 ]; then
        exit 1  # Some processes couldn't be killed
    else
        exit 0  # Success
    fi
}

# Handle script being interrupted
trap 'log_warn "Script interrupted"; exit 130' INT TERM

# Run main function
main "$@"
