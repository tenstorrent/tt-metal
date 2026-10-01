#!/usr/bin/env bash
# Usage: get-target-tools.sh <bake-target-or-group>
# Prints space-separated tool names derived from docker-bake.hcl.
#
# For consumer targets (ci-build-light, ci-build, ci-test-light, ci-test, dev-light, dev, basic-dev, manylinux, evaluation):
#   Extracts context keys ending in "-layer", strips the suffix.
#
# For tool groups (tools):
#   Extracts group target names directly.
#
# Examples:
#   get-target-tools.sh ci-build-light # => ccache mold doxygen clangbuildanalyzer gdb cmake yq zstd sfpi openmpi
#   get-target-tools.sh dev-light     # => ccache mold doxygen clangbuildanalyzer gdb cmake yq zstd sfpi openmpi
#   get-target-tools.sh dev           # => ccache mold doxygen clangbuildanalyzer gdb cmake yq zstd sfpi openmpi
#   get-target-tools.sh basic-dev     # => ccache cmake openmpi sfpi
#   get-target-tools.sh manylinux     # => ccache mold openmpi sfpi zstd
#   get-target-tools.sh tools         # => ccache clangbuildanalyzer cmake doxygen gdb mold openmpi sfpi yq zstd
set -euo pipefail

TARGET="${1:?Usage: $0 <bake-target-or-group>}"

BAKE_JSON=$(docker buildx bake -f dockerfile/docker-bake.hcl --print "$TARGET" 2>/dev/null)

# Check if the output has a group key for this target (i.e. it's a group, not a consumer target)
if echo "$BAKE_JSON" | jq -e ".group.\"${TARGET}\"" > /dev/null 2>&1; then
  # It's a group — return the group's target names
  echo "$BAKE_JSON" \
    | jq -r ".group.\"${TARGET}\".targets[]" \
    | sort \
    | tr '\n' ' ' \
    | sed 's/ $//'
else
  # It's a consumer target — extract context keys ending in "-layer"
  echo "$BAKE_JSON" \
    | jq -r ".target.\"${TARGET}\".contexts // {} | keys[] | select(endswith(\"-layer\") and (startswith(\"ci-\") | not)) | sub(\"-layer$\";\"\")" \
    | sort \
    | tr '\n' ' ' \
    | sed 's/ $//'
fi
