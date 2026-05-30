#!/bin/bash
# Compute tool image tags and check existence
#
# Usage: compute-tool-data.sh <repo> [--force-rebuild] [--check-exists]
#
# Arguments:
#   repo            GitHub repository (e.g., "owner/repo")
#   --force-rebuild Treat all images as missing (optional)
#   --check-exists  Check if images exist in registry (optional, default: true)
#
# Output: JSON object with tool_tags, any_missing, and per-tool existence flags
#
# Example:
#   compute-tool-data.sh "tenstorrent/tt-metal" --check-exists

set -euo pipefail

usage() {
    echo "Usage: $0 <repo> [--force-rebuild] [--check-exists]" >&2
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

REPO="$1"
shift

FORCE_REBUILD=false
CHECK_EXISTS=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force-rebuild)
            FORCE_REBUILD=true
            shift
            ;;
        --check-exists)
            CHECK_EXISTS=true
            shift
            ;;
        --no-check-exists)
            CHECK_EXISTS=false
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
done

# Compute tool tags using existing script (always produces canonical ghcr.io tags)
TOOL_TAGS=$(.github/scripts/compute-tool-tags.sh "$REPO")

# Derive tool list from docker-bake.hcl (single source of truth)
TOOLS=$(docker buildx bake -f dockerfile/docker-bake.hcl --print tools 2>/dev/null \
  | jq -r '.group.tools.targets[]' | sort | tr '\n' ' ' | sed 's/ $//')

# Check existence for each tool (parallel to avoid serial network latency)
ANY_MISSING=false
declare -A EXISTS
declare -A PIDS
TMPDIR_EXISTS=$(mktemp -d)

for tool in $TOOLS; do
    tag=$(echo "$TOOL_TAGS" | jq -r ".\"${tool}-tag\"")

    if [ "$FORCE_REBUILD" = "true" ]; then
        EXISTS[$tool]=false
        ANY_MISSING=true
    elif [ "$CHECK_EXISTS" = "true" ]; then
        # Fire all manifest inspects in parallel; results written to temp files
        ( docker manifest inspect "$tag" > /dev/null 2>&1 && echo true || echo false ) \
            > "${TMPDIR_EXISTS}/${tool}" &
        PIDS[$tool]=$!
    else
        EXISTS[$tool]=unknown
    fi
done

# Collect parallel results
if [ "$CHECK_EXISTS" = "true" ] && [ "$FORCE_REBUILD" = "false" ]; then
    for tool in $TOOLS; do
        wait "${PIDS[$tool]}"
        EXISTS[$tool]=$(cat "${TMPDIR_EXISTS}/${tool}")
        [ "${EXISTS[$tool]}" = "false" ] && ANY_MISSING=true
    done
fi

rm -rf "$TMPDIR_EXISTS"

# Build output JSON dynamically from derived tool list
# Start with base fields, then add per-tool existence flags
JQ_ARGS=(--argjson tool_tags "$TOOL_TAGS" --arg any_missing "$ANY_MISSING")
EXISTS_OBJ="{}"
for tool in $TOOLS; do
    JQ_ARGS+=(--arg "${tool}_exists" "${EXISTS[$tool]}")
    EXISTS_OBJ=$(echo "$EXISTS_OBJ" | jq --arg k "${tool}_exists" --arg v "${EXISTS[$tool]}" '. + {($k): $v}')
done

jq -n \
    --argjson tool_tags "$TOOL_TAGS" \
    --arg any_missing "$ANY_MISSING" \
    --argjson exists_obj "$EXISTS_OBJ" \
    '{tool_tags: $tool_tags, any_missing: $any_missing} + $exists_obj'
