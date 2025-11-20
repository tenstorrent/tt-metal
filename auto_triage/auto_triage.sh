#!/bin/bash
#
# Full triage driver: wipes old data/logs, finds boundaries, then invokes OpenCode.
# Usage:
#   ./auto_triage.sh <workflow_name> <subjob_name>
# Example:
#   ./auto_triage.sh galaxy-quick quick-wh-glx-quick

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <workflow_name> <subjob_name>" >&2
    exit 1
fi

WORKFLOW="$1"
SUBJOB="$2"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT}/auto_triage/data"
LOGS_DIR="${ROOT}/auto_triage/logs"
FIND_SCRIPT="${ROOT}/auto_triage/find_boundaries.sh"

if [ ! -x "$FIND_SCRIPT" ]; then
    echo "Error: $FIND_SCRIPT not found or not executable." >&2
    exit 1
fi

echo "=== Cleaning auto_triage/data and auto_triage/logs ==="
rm -rf "$DATA_DIR" "$LOGS_DIR"
mkdir -p "$DATA_DIR" "$LOGS_DIR"

cd "$ROOT"
echo "=== Running find_boundaries.sh for workflow '${WORKFLOW}' / job '${SUBJOB}' ==="
"$FIND_SCRIPT" "$WORKFLOW" "$SUBJOB"

if ! command -v opencode >/dev/null 2>&1; then
    echo "Error: opencode CLI is required but not found in PATH." >&2
    exit 1
fi

PROMPT="complete the instructions defined in auto_triage/instructions_for_opencode.txt for workflow '${WORKFLOW}' and job '${SUBJOB}'. do not wait for approval as this is a test run for a CI environment where you will not have access to a user"

echo "=== Launching OpenCode ==="
opencode run -m opencode/gemini-3-pro "$PROMPT"
