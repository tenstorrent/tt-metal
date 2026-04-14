#!/bin/bash
# Initialize execution log for an operation
# Usage: init_log.sh <operation> <agent_name>

set -e

OPERATION="$1"
AGENT_NAME="$2"

if [[ -z "$OPERATION" || -z "$AGENT_NAME" ]]; then
    echo "Usage: init_log.sh <operation> <agent_name>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEGEN_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_FILE="$CODEGEN_DIR/artifacts/${OPERATION}_log.md"

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Create or append to log file
cat >> "$LOG_FILE" << EOF

---

## ${AGENT_NAME}
**Started**: ${TIMESTAMP}

### Events
EOF

echo "Log initialized: $LOG_FILE"
