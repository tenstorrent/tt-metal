#!/usr/bin/env bash
# ensure_bh_links.sh - Ensure Blackhole ethernet links are online
# Usage: ensure_bh_links.sh [OPTIONS]
#
# Port of .github/actions/ensure-bh-links-online/action.yml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

MAX_RETRIES=10
BACKOFF=5
HEALTH_CHECK_BIN="${HEALTH_CHECK_BIN:-./build/test/tt_metal/tt_fabric/test_system_health}"
HEALTH_CHECK_EXTRA_ARGS=""

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Perform tt-smi reset and health-check loop to ensure Blackhole ethernet
links are online. Needed due to SYS-1634 / BH-84.

Options:
  --max-retries N       Maximum health-check attempts (default: 10)
  --backoff SECONDS     Sleep between attempts (default: 5)
  --health-check-args   Extra arguments for the health check binary
  -h, --help            Show this help message

Environment:
  HEALTH_CHECK_BIN      Path to system health binary
                        (default: ./build/test/tt_metal/tt_fabric/test_system_health)
EOF
    exit "${1:-0}"
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-retries)      MAX_RETRIES="$2";          shift 2 ;;
        --backoff)          BACKOFF="$2";               shift 2 ;;
        --health-check-args) HEALTH_CHECK_EXTRA_ARGS="$2"; shift 2 ;;
        -h|--help)          usage 0 ;;
        *)                  log_error "Unknown option: $1"; usage 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Health check loop
# ---------------------------------------------------------------------------

require_cmd tt-smi

log_info "Starting Blackhole link health check (max ${MAX_RETRIES} attempts, ${BACKOFF}s backoff)"

for (( i = 1; i <= MAX_RETRIES; i++ )); do
    log_info "Health check attempt ${i}/${MAX_RETRIES}"

    # Reset devices then run the health check binary
    # shellcheck disable=SC2086
    if tt-smi -r >/dev/null 2>&1 && ${HEALTH_CHECK_BIN} ${HEALTH_CHECK_EXTRA_ARGS}; then
        log_info "Health checks passed on attempt ${i}"
        exit 0
    fi

    if (( i == MAX_RETRIES )); then
        log_error "Health checks failed after ${MAX_RETRIES} attempts"
        exit 1
    fi

    log_warn "Health checks failed, retrying in ${BACKOFF}s..."
    sleep "$BACKOFF"
done
