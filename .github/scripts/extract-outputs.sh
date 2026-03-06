#!/bin/bash
# Extract fields from JSON and output in GitHub Actions format
#
# Usage: extract-outputs.sh <json_data> <output_name>...
#
# Output names are automatically converted to jq selectors:
#   - Kebab-case is converted to snake_case
#   - A leading '.' is prepended
#   Example: "tool-tags" becomes ".tool_tags"
#
# For custom mappings, use "output_name:jq_selector" format:
#   Example: "basic-ttnn-runtime-exists:.basic_ttnn_exists"
#
# Examples:
#   # Simple (auto-derived selectors)
#   extract-outputs.sh "$JSON_DATA" tool-tags any-missing ccache-exists
#
#   # Custom mapping when names don't match
#   extract-outputs.sh "$JSON_DATA" "basic-ttnn-runtime-tag:.basic_ttnn_tag"
#
# Supports multi-line strings using heredoc format automatically.
# JSON can be passed via stdin using "-" as the first argument.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <json_data | -> <output_name>..." >&2
    echo "  Output names auto-convert to jq selectors (kebab → snake_case)" >&2
    echo "  Or use explicit mapping: \"output_name:jq_selector\"" >&2
    exit 1
fi

JSON_DATA="$1"
shift

# Read from stdin if "-" is passed
if [[ "$JSON_DATA" == "-" ]]; then
    JSON_DATA=$(cat)
fi

for arg in "$@"; do
    # Check if explicit mapping is provided (contains ':')
    if [[ "$arg" == *":"* ]]; then
        output_name="${arg%%:*}"
        jq_selector="${arg#*:}"
    else
        # Auto-derive: convert kebab-case to snake_case and prepend '.'
        output_name="$arg"
        jq_selector=".${arg//-/_}"
    fi

    if [[ -z "$output_name" ]]; then
        echo "Error: Empty output name in '$arg'" >&2
        exit 1
    fi

    # Extract value using jq
    value=$(echo "$JSON_DATA" | jq -r "$jq_selector" 2>/dev/null || echo "")

    # Check if value contains newlines (multi-line string)
    if [[ "$value" == *$'\n'* ]]; then
        # Use heredoc format for multi-line values
        delimiter="EOF_${output_name}"
        # Sanitize delimiter (replace non-alphanumeric with underscore)
        delimiter=$(echo "$delimiter" | sed 's/[^a-zA-Z0-9_]/_/g')
        {
            echo "${output_name}<<${delimiter}"
            echo "$value"
            echo "${delimiter}"
        } >> "$GITHUB_OUTPUT"
    else
        # Single line value
        echo "${output_name}=${value}" >> "$GITHUB_OUTPUT"
    fi
done
