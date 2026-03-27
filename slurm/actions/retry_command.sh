#!/usr/bin/env bash
# retry_command.sh - CLI wrapper around lib/retry.sh retry_command()
# Usage: retry_command.sh --command "CMD" --timeout SECONDS [OPTIONS]
#
# Equivalent to .github/actions/retry-command/action.yml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib retry

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

COMMAND=""
TIMEOUT=""
BACKOFF=5
MAX_RETRIES=3

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $(basename "$0") --command "CMD" --timeout SECONDS [OPTIONS]

Run a command with retries and exponential backoff on failure.

Required:
  --command "CMD"      Command to execute (passed to bash -c)
  --timeout SECONDS    Timeout per attempt in seconds

Options:
  --backoff SECONDS    Initial backoff between retries (default: 5, doubles each retry)
  --max-retries N      Maximum number of attempts (default: 3)
  -h, --help           Show this help message
EOF
    exit "${1:-0}"
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --command)     COMMAND="$2";     shift 2 ;;
        --timeout)     TIMEOUT="$2";     shift 2 ;;
        --backoff)     BACKOFF="$2";     shift 2 ;;
        --max-retries) MAX_RETRIES="$2"; shift 2 ;;
        -h|--help)     usage 0 ;;
        *)             log_error "Unknown option: $1"; usage 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if [[ -z "$COMMAND" ]]; then
    log_error "--command is required"
    usage 1
fi

if [[ -z "$TIMEOUT" ]]; then
    log_error "--timeout is required"
    usage 1
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

retry_command "$COMMAND" "$TIMEOUT" "$BACKOFF" "$MAX_RETRIES"
