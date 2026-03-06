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

# Compute tool tags using existing script
TOOL_TAGS=$(HARBOR_PREFIX="" .github/scripts/compute-tool-tags.sh "$REPO")

# Tool list
TOOLS="ccache mold doxygen cba gdb cmake yq sfpi openmpi"

# Check existence for each tool
ANY_MISSING=false
declare -A EXISTS

for tool in $TOOLS; do
    tag=$(echo "$TOOL_TAGS" | jq -r ".\"${tool}-tag\"")

    if [ "$FORCE_REBUILD" = "true" ]; then
        EXISTS[$tool]=false
        ANY_MISSING=true
    elif [ "$CHECK_EXISTS" = "true" ]; then
        if docker manifest inspect "$tag" > /dev/null 2>&1; then
            EXISTS[$tool]=true
        else
            EXISTS[$tool]=false
            ANY_MISSING=true
        fi
    else
        EXISTS[$tool]=unknown
    fi
done

# Output JSON with tool_tags as nested object, not string
jq -n \
    --argjson tool_tags "$TOOL_TAGS" \
    --arg any_missing "$ANY_MISSING" \
    --arg ccache_exists "${EXISTS[ccache]}" \
    --arg mold_exists "${EXISTS[mold]}" \
    --arg doxygen_exists "${EXISTS[doxygen]}" \
    --arg cba_exists "${EXISTS[cba]}" \
    --arg gdb_exists "${EXISTS[gdb]}" \
    --arg cmake_exists "${EXISTS[cmake]}" \
    --arg yq_exists "${EXISTS[yq]}" \
    --arg sfpi_exists "${EXISTS[sfpi]}" \
    --arg openmpi_exists "${EXISTS[openmpi]}" \
    '{
        tool_tags: $tool_tags,
        any_missing: $any_missing,
        ccache_exists: $ccache_exists,
        mold_exists: $mold_exists,
        doxygen_exists: $doxygen_exists,
        cba_exists: $cba_exists,
        gdb_exists: $gdb_exists,
        cmake_exists: $cmake_exists,
        yq_exists: $yq_exists,
        sfpi_exists: $sfpi_exists,
        openmpi_exists: $openmpi_exists
    }'
