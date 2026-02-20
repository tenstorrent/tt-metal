#!/bin/bash
# Set logging configuration for an operation
# Usage: set_logging_config.sh <operation_path> [--enable|--disable] [--verbosity=<level>]
#
# Examples:
#   set_logging_config.sh ttnn/cpp/ttnn/operations/reduction/my_op --enable
#   set_logging_config.sh ttnn/cpp/ttnn/operations/reduction/my_op --disable
#   set_logging_config.sh ttnn/cpp/ttnn/operations/reduction/my_op --enable --verbosity=detailed
#
# This creates/updates: {operation_path}/agent_logs/logging_config.json

set -e

OPERATION_PATH="$1"
shift

if [[ -z "$OPERATION_PATH" ]]; then
    echo "Usage: set_logging_config.sh <operation_path> [--enable|--disable] [--verbosity=<level>]" >&2
    exit 1
fi

# Parse arguments
ENABLED="false"
VERBOSITY="normal"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --enable)
            ENABLED="true"
            shift
            ;;
        --disable)
            ENABLED="false"
            shift
            ;;
        --verbosity=*)
            VERBOSITY="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# Ensure directory exists
LOG_DIR="${OPERATION_PATH}/agent_logs"
mkdir -p "$LOG_DIR"

CONFIG_FILE="${LOG_DIR}/logging_config.json"

# Write config
cat > "$CONFIG_FILE" << EOF
{
  "breadcrumbs_enabled": $ENABLED,
  "verbosity": "$VERBOSITY",
  "created_at": "$(date -Iseconds)"
}
EOF

echo "Logging config written to $CONFIG_FILE"
echo "  breadcrumbs_enabled: $ENABLED"
echo "  verbosity: $VERBOSITY"
