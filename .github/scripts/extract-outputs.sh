#!/bin/bash
# Extract fields from JSON and output in GitHub Actions format
#
# Usage: extract-outputs.sh <json_data> <field_mapping>...
#
# Field mapping format: "output_name:jq_selector"
#   - output_name: Name of the GitHub Actions output
#   - jq_selector: jq selector for extracting the value from JSON
#
# Example:
#   extract-outputs.sh "$JSON_DATA" \
#     "tool-tags:.tool_tags" \
#     "any-missing:.any_missing" \
#     "ccache-exists:.ccache_exists"
#
# Supports multi-line strings using heredoc format automatically.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <json_data> <field_mapping>..." >&2
    echo "  Field mapping format: \"output_name:jq_selector\"" >&2
    exit 1
fi

JSON_DATA="$1"
shift

for mapping in "$@"; do
    # Split mapping into output name and jq selector
    output_name="${mapping%%:*}"
    jq_selector="${mapping#*:}"

    if [[ -z "$output_name" || -z "$jq_selector" ]]; then
        echo "Error: Invalid field mapping '$mapping'. Expected format: 'output_name:jq_selector'" >&2
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
