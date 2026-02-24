#!/bin/bash
# Check if breadcrumb logging is enabled for an operation
# Usage: check_logging_enabled.sh <operation_path>
#
# Returns:
#   exit 0 if logging enabled (breadcrumbs_enabled: true)
#   exit 1 if logging disabled or config not found
#
# Example:
#   if .claude/scripts/logging/check_logging_enabled.sh ttnn/cpp/ttnn/operations/reduction/my_op; then
#       echo "Logging enabled"
#   else
#       echo "Logging disabled"
#   fi

OPERATION_PATH="$1"

if [[ -z "$OPERATION_PATH" ]]; then
    echo "Usage: check_logging_enabled.sh <operation_path>" >&2
    exit 1
fi

CONFIG_FILE="${OPERATION_PATH}/agent_logs/logging_config.json"

# If config file doesn't exist, logging is disabled
if [[ ! -f "$CONFIG_FILE" ]]; then
    exit 1
fi

# Check if breadcrumbs_enabled is true
if command -v jq &> /dev/null; then
    # Use jq if available (more robust)
    jq -e '.breadcrumbs_enabled == true' "$CONFIG_FILE" > /dev/null 2>&1
    exit $?
else
    # Fallback: grep for the field (less robust but works)
    grep -q '"breadcrumbs_enabled"[[:space:]]*:[[:space:]]*true' "$CONFIG_FILE" 2>/dev/null
    exit $?
fi
