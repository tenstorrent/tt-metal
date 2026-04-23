#!/bin/bash
# Enable or disable logging for an operation
# Usage: set_logging.sh <operation> --enable|--disable

set -e

OPERATION="$1"
ACTION="$2"

if [[ -z "$OPERATION" || -z "$ACTION" ]]; then
    echo "Usage: set_logging.sh <operation> --enable|--disable"
    echo "Example: set_logging.sh sigmoid --enable"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEGEN_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="$CODEGEN_DIR/artifacts/.logging_config"

mkdir -p "$CODEGEN_DIR/artifacts"

case "$ACTION" in
    --enable)
        echo "$OPERATION" >> "$CONFIG_FILE"
        # Remove duplicates
        sort -u "$CONFIG_FILE" -o "$CONFIG_FILE"
        echo "Logging ENABLED for: $OPERATION"
        echo "Logs will be written to: codegen/artifacts/${OPERATION}_log.md"
        ;;
    --disable)
        if [[ -f "$CONFIG_FILE" ]]; then
            grep -v "^${OPERATION}$" "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" || true
            mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"
        fi
        echo "Logging DISABLED for: $OPERATION"
        ;;
    *)
        echo "Unknown action: $ACTION"
        echo "Use --enable or --disable"
        exit 1
        ;;
esac
