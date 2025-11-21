#!/bin/bash
#
# Full triage driver: wipes old data/logs, finds boundaries, then invokes OpenCode.
# Usage:
#   ./auto_triage.sh <workflow_name> <subjob_name> <model>
# Example:
#   ./auto_triage.sh galaxy-quick quick-wh-glx-quick openai/gpt-5.1-codex-mini

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: $0 <workflow_name> <subjob_name> <model>" >&2
    exit 1
fi

WORKFLOW="$1"
SUBJOB="$2"
MODEL="$3"

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
rm -rf "$ROOT/auto_triage/output"
mkdir -p "$ROOT/auto_triage/output"

cd "$ROOT"
echo "=== Running find_boundaries.sh for workflow '${WORKFLOW}' / job '${SUBJOB}' ==="
"$FIND_SCRIPT" "$WORKFLOW" "$SUBJOB"

if ! command -v opencode >/dev/null 2>&1; then
    echo "Error: opencode CLI is required but not found in PATH." >&2
    exit 1
fi

INSTRUCTIONS_FILE="${ROOT}/auto_triage/instructions_for_opencode.txt"
if [ ! -f "$INSTRUCTIONS_FILE" ]; then
    echo "Error: ${INSTRUCTIONS_FILE} not found." >&2
    exit 1
fi

read -r -d '' PROMPT <<EOF || true
You are operating in a CI environment with no interactive approval. Complete the following instructions for workflow '${WORKFLOW}' and job '${SUBJOB}':

$(cat "$INSTRUCTIONS_FILE")
EOF

echo "=== Launching OpenCode (model: ${MODEL}) ==="
opencode run -m "$MODEL" "$PROMPT"
