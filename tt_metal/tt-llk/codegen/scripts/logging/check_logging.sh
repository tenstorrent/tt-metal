#!/bin/bash
# Check if logging is enabled for an operation
# Usage: check_logging.sh <operation>
# Returns: exit 0 if enabled, exit 1 if disabled

OPERATION="$1"

if [[ -z "$OPERATION" ]]; then
    echo "Usage: check_logging.sh <operation>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEGEN_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="$CODEGEN_DIR/artifacts/.logging_config"

if [[ -f "$CONFIG_FILE" ]] && grep -q "^${OPERATION}$" "$CONFIG_FILE"; then
    echo "enabled"
    exit 0
else
    echo "disabled"
    exit 1
fi
