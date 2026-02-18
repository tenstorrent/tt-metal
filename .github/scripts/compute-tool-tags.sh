#!/bin/bash
# Compute content-addressed tool image tags for all tools in Dockerfile.tools.
# Single source of truth for tag computation; used by build-docker-tools.yaml
# and build-evaluation-image.yaml to avoid drift.
#
# Usage: compute-tool-tags.sh [REPOSITORY]
#   REPOSITORY defaults to $GITHUB_REPOSITORY (e.g. tenstorrent/tt-metal)
#
# Outputs: JSON object to stdout with keys ccache-tag, mold-tag, etc.
# Example: {"ccache-tag":"ghcr.io/tenstorrent/tt-metal/tt-metalium/tools/ccache:4.10.2-abc12345",...}

set -euo pipefail

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
CCACHE_HASH=$(cat dockerfile/scripts/install-ccache.sh | sha1sum | cut -d' ' -f1 | head -c 8)
MOLD_HASH=$(cat dockerfile/scripts/install-mold.sh | sha1sum | cut -d' ' -f1 | head -c 8)
DOXYGEN_HASH=$(cat dockerfile/scripts/install-doxygen.sh | sha1sum | cut -d' ' -f1 | head -c 8)
CBA_HASH=$(cat dockerfile/scripts/install-clangbuildanalyzer.sh | sha1sum | cut -d' ' -f1 | head -c 8)
GDB_HASH=$(cat dockerfile/scripts/install-gdb.sh | sha1sum | cut -d' ' -f1 | head -c 8)
CMAKE_HASH=$(cat dockerfile/scripts/install-cmake.sh | sha1sum | cut -d' ' -f1 | head -c 8)
YQ_HASH=$(cat dockerfile/scripts/install-yq.sh | sha1sum | cut -d' ' -f1 | head -c 8)
SFPI_HASH=$(cat dockerfile/scripts/install-sfpi.sh tt_metal/sfpi-version | sha1sum | cut -d' ' -f1 | head -c 8)
OPENMPI_HASH=$(cat dockerfile/scripts/install-openmpi.sh | sha1sum | cut -d' ' -f1 | head -c 8)

# Generate tags: ghcr.io/<repo>/tt-metalium/tools/<tool>:<version>-<hash>
BASE="ghcr.io/${REPO}/tt-metalium/tools"

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
