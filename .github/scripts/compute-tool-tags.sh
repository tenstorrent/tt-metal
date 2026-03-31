#!/bin/bash
# Compute content-addressed tool image tags for all tools in Dockerfile.tools.
# Single source of truth for tag computation; used by build-docker-tools.yaml
# and build-evaluation-image.yaml to avoid drift.
#
# Usage: compute-tool-tags.sh [REPOSITORY]
#   REPOSITORY defaults to $GITHUB_REPOSITORY (e.g. tenstorrent/tt-metal)
#
# Outputs: JSON object to stdout with keys ccache-tag, mold-tag, etc.
# Example: {"ccache-tag":"ghcr.io/tenstorrent/tt-metal/tt-metalium/tools/ccache:4.10.2-abc123456789ab",...}
#
# Environment variables:
#   HARBOR_PREFIX - Optional prefix for Harbor pull-through cache (e.g., "harbor.ci.tenstorrent.net/")

set -euo pipefail

# Optional Harbor pull-through cache prefix (default: empty string)
HARBOR_PREFIX="${HARBOR_PREFIX:-}"

REPO="${1:-${GITHUB_REPOSITORY:?GITHUB_REPOSITORY or repository argument required}}"

# Extract versions from Dockerfile.tools (head -1 so multiple ARG lines don't produce multiline output)
CCACHE_VERSION=$(grep -E "^ARG CCACHE_VERSION=" dockerfile/Dockerfile.tools | head -1 | cut -d= -f2)
MOLD_VERSION=$(grep -E "^ARG MOLD_VERSION=" dockerfile/Dockerfile.tools | head -1 | cut -d= -f2)
DOXYGEN_VERSION=$(grep -E "^ARG DOXYGEN_VERSION=" dockerfile/Dockerfile.tools | head -1 | cut -d= -f2)
CBA_VERSION=$(grep -E "^ARG CBA_VERSION=" dockerfile/Dockerfile.tools | head -1 | cut -d= -f2)
GDB_VERSION=$(grep -E "^ARG GDB_VERSION=" dockerfile/Dockerfile.tools | head -1 | cut -d= -f2)
CMAKE_VERSION=$(grep -E "^ARG CMAKE_VERSION=" dockerfile/Dockerfile.tools | head -1 | cut -d= -f2)
YQ_VERSION=$(grep -E "^ARG YQ_VERSION=" dockerfile/Dockerfile.tools | head -1 | cut -d= -f2)
OMPI_TAG=$(grep -E "^ARG OMPI_TAG=" dockerfile/Dockerfile.tools | head -1 | cut -d= -f2)
SFPI_VERSION=$(grep -E "^sfpi_version=" tt_metal/sfpi-version | cut -d"'" -f2)

# Compute hashes for each tool (version + install script)
for tool in ccache mold doxygen cba gdb cmake yq; do
    script_name="install-${tool}.sh"
    [[ "$tool" == "cba" ]] && script_name="install-clangbuildanalyzer.sh"
    hash_var="${tool^^}_HASH"
    declare "$hash_var=$(cat "dockerfile/scripts/${script_name}" | sha1sum | cut -d' ' -f1 | head -c 12)"
done

# Handle special cases (sfpi and openmpi) separately
SFPI_HASH=$(cat dockerfile/scripts/install-sfpi.sh tt_metal/sfpi-version | sha1sum | cut -d' ' -f1 | head -c 12)
OPENMPI_HASH=$(cat dockerfile/scripts/install-openmpi.sh | sha1sum | cut -d' ' -f1 | head -c 12)

# Generate tags: <harbor-prefix>ghcr.io/<repo>/tt-metalium/tools/<tool>:<version>-<hash>
BASE="${HARBOR_PREFIX}ghcr.io/${REPO}/tt-metalium/tools"

jq -n \
  --arg ccache "${BASE}/ccache:${CCACHE_VERSION}-${CCACHE_HASH}" \
  --arg mold "${BASE}/mold:${MOLD_VERSION}-${MOLD_HASH}" \
  --arg doxygen "${BASE}/doxygen:${DOXYGEN_VERSION}-${DOXYGEN_HASH}" \
  --arg cba "${BASE}/cba:${CBA_VERSION}-${CBA_HASH}" \
  --arg gdb "${BASE}/gdb:${GDB_VERSION}-${GDB_HASH}" \
  --arg cmake "${BASE}/cmake:${CMAKE_VERSION}-${CMAKE_HASH}" \
  --arg yq "${BASE}/yq:${YQ_VERSION}-${YQ_HASH}" \
  --arg sfpi "${BASE}/sfpi:${SFPI_VERSION}-${SFPI_HASH}" \
  --arg openmpi "${BASE}/openmpi:${OMPI_TAG}-${OPENMPI_HASH}" \
  '{
    "ccache-tag": $ccache,
    "mold-tag": $mold,
    "doxygen-tag": $doxygen,
    "cba-tag": $cba,
    "gdb-tag": $gdb,
    "cmake-tag": $cmake,
    "yq-tag": $yq,
    "sfpi-tag": $sfpi,
    "openmpi-tag": $openmpi
  }'
