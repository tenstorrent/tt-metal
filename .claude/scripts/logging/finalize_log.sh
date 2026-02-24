#!/bin/bash
# Finalize agent execution log
# This script is called by the Stop hook to remind the agent to write the final log
#
# Usage: finalize_log.sh <operation_path> <agent_name>
#
# This script checks if a structured log exists. If not, it outputs a reminder.
# The actual log writing is done by the agent using the template.

set -e

OPERATION_PATH="$1"
AGENT_NAME="$2"

if [[ -z "$OPERATION_PATH" || -z "$AGENT_NAME" ]]; then
    echo "Usage: finalize_log.sh <operation_path> <agent_name>" >&2
    exit 1
fi

LOG_DIR="${OPERATION_PATH}/agent_logs"
BREADCRUMB_FILE="${LOG_DIR}/${AGENT_NAME}_breadcrumbs.jsonl"
EXECUTION_LOG="${LOG_DIR}/${AGENT_NAME}_execution_log.md"

# Check if breadcrumbs exist but execution log doesn't
if [[ -f "$BREADCRUMB_FILE" && ! -f "$EXECUTION_LOG" ]]; then
    echo "================================================================"
    echo "REMINDER: Execution log not yet written!"
    echo ""
    echo "Breadcrumbs file: $BREADCRUMB_FILE"
    echo "Expected log: $EXECUTION_LOG"
    echo ""
    echo "Before completing, you must:"
    echo "1. Read the log template: .claude/references/agent-log-template.md"
    echo "2. Read your breadcrumbs: $BREADCRUMB_FILE"
    echo "3. Write the structured log to: $EXECUTION_LOG"
    echo "================================================================"
    exit 0
fi

if [[ -f "$EXECUTION_LOG" ]]; then
    echo "Execution log exists: $EXECUTION_LOG"

    # Optionally clean up breadcrumbs after successful log creation
    # Uncomment the following line to enable auto-cleanup:
    # rm -f "$BREADCRUMB_FILE"
fi
