#!/usr/bin/env bash
# cleanup.sh - CLI wrapper for workspace cleanup
# Usage: cleanup.sh [--path PATH]
#
# Equivalent to .github/actions/cleanup/action.yml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Remove the workspace directory. Cleans up root-owned files left behind
by Docker containers on non-ephemeral runners.

Options:
  --path PATH    Directory to remove (default: \$WORKSPACE or /work)
  -h, --help     Show this help message

Environment:
  WORKSPACE      Default directory to clean if --path is not given
EOF
    exit "${1:-0}"
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

CLEANUP_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --path)    CLEANUP_PATH="$2"; shift 2 ;;
        -h|--help) usage 0 ;;
        *)         log_error "Unknown option: $1"; usage 1 ;;
    esac
done

CLEANUP_PATH="${CLEANUP_PATH:-${WORKSPACE:-/work}}"

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

if [[ ! -d "$CLEANUP_PATH" ]]; then
    log_warn "Directory does not exist, nothing to clean: ${CLEANUP_PATH}"
    exit 0
fi

log_info "Removing workspace: ${CLEANUP_PATH}"
ls -al "$CLEANUP_PATH" 2>/dev/null || true
rm -rf "$CLEANUP_PATH"
log_info "Workspace removed"
ls -al "$(dirname "$CLEANUP_PATH")" 2>/dev/null || true
