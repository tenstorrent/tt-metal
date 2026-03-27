#!/usr/bin/env bash
# retry.sh - Retry-with-backoff wrapper for Slurm CI
# Port of .github/actions/retry-command.
# Parameters can be passed as arguments or via env vars:
#   RETRY_TIMEOUT  - timeout in seconds (default: 300)
#   RETRY_BACKOFF  - initial backoff in seconds (default: 5)
#   RETRY_MAX      - maximum retry attempts (default: 3)

set -euo pipefail

[[ -n "${_SLURM_CI_RETRY_SH:-}" ]] && return 0
_SLURM_CI_RETRY_SH=1

# ---------------------------------------------------------------------------
# retry_command - Run a command with timeout, retrying on failure with backoff
# ---------------------------------------------------------------------------
# Usage: retry_command <command> [timeout_seconds] [backoff_seconds] [max_retries]
#   Returns the command's exit code (0 on success, 1 after exhausting retries).
retry_command() {
    local command="$1"
    local timeout_seconds="${2:-${RETRY_TIMEOUT:-300}}"
    local backoff_seconds="${3:-${RETRY_BACKOFF:-5}}"
    local max_retries="${4:-${RETRY_MAX:-3}}"

    local count=0
    local rc=0

    while (( count < max_retries )); do
        count=$((count + 1))
        log_info "retry_command: attempt ${count}/${max_retries} (timeout=${timeout_seconds}s)"

        rc=0
        timeout "${timeout_seconds}" bash -c "${command}" && return 0 || rc=$?

        if (( rc == 124 )); then
            log_warn "retry_command: command timed out after ${timeout_seconds}s (attempt ${count})"
        else
            log_warn "retry_command: command exited with code ${rc} (attempt ${count})"
        fi

        if (( count < max_retries )); then
            log_info "retry_command: backing off ${backoff_seconds}s before next attempt"
            sleep "${backoff_seconds}"
        fi
    done

    log_error "retry_command: failed after ${max_retries} attempts: ${command}"
    return "${rc}"
}

# ---------------------------------------------------------------------------
# retry_with_exponential_backoff - Retry with doubling backoff between attempts
# ---------------------------------------------------------------------------
# Usage: retry_with_exponential_backoff <command> [max_retries] [initial_backoff]
#   Backoff doubles after each failure: initial, initial*2, initial*4, ...
#   No timeout is applied to individual attempts; use retry_command if you
#   need per-attempt timeouts.
retry_with_exponential_backoff() {
    local command="$1"
    local max_retries="${2:-${RETRY_MAX:-3}}"
    local backoff="${3:-${RETRY_BACKOFF:-5}}"

    local count=0
    local rc=0

    while (( count < max_retries )); do
        count=$((count + 1))
        log_info "retry_exp_backoff: attempt ${count}/${max_retries}"

        rc=0
        bash -c "${command}" && return 0 || rc=$?

        log_warn "retry_exp_backoff: command exited with code ${rc} (attempt ${count})"

        if (( count < max_retries )); then
            log_info "retry_exp_backoff: backing off ${backoff}s before next attempt"
            sleep "${backoff}"
            backoff=$((backoff * 2))
        fi
    done

    log_error "retry_exp_backoff: failed after ${max_retries} attempts: ${command}"
    return "${rc}"
}
