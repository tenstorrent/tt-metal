#!/bin/bash
# Append event to execution log
# Usage: append_log.sh <operation> <event_type> <message>

set -e

OPERATION="$1"
EVENT_TYPE="$2"
MESSAGE="$3"

if [[ -z "$OPERATION" || -z "$EVENT_TYPE" || -z "$MESSAGE" ]]; then
    echo "Usage: append_log.sh <operation> <event_type> <message>"
    echo "Event types: action, result, error, hypothesis, recovery, complete"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEGEN_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_FILE="$CODEGEN_DIR/artifacts/${OPERATION}_log.md"

TIMESTAMP=$(date '+%H:%M:%S')

# Format based on event type
case "$EVENT_TYPE" in
    action)
        echo "- [$TIMESTAMP] **Action**: $MESSAGE" >> "$LOG_FILE"
        ;;
    result)
        echo "- [$TIMESTAMP] **Result**: $MESSAGE" >> "$LOG_FILE"
        ;;
    error)
        echo "- [$TIMESTAMP] **Error**: $MESSAGE" >> "$LOG_FILE"
        ;;
    hypothesis)
        echo "- [$TIMESTAMP] **Hypothesis**: $MESSAGE" >> "$LOG_FILE"
        ;;
    recovery)
        echo "- [$TIMESTAMP] **Recovery**: $MESSAGE" >> "$LOG_FILE"
        ;;
    complete)
        echo "" >> "$LOG_FILE"
        echo "**Completed**: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
        echo "**Status**: $MESSAGE" >> "$LOG_FILE"
        ;;
    *)
        echo "- [$TIMESTAMP] [$EVENT_TYPE]: $MESSAGE" >> "$LOG_FILE"
        ;;
esac
