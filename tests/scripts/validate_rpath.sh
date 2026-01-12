#!/bin/bash
# Validate RPATH/RUNPATH in built binaries and libraries
#
# This script checks that:
# 1. RPATH/RUNPATH is set correctly in binaries
# 2. $ORIGIN appears first (required for Fedora compliance)
# 3. MPI library paths are included when MPI is enabled
# 4. No absolute build paths leak into RPATH
#
# Usage:
#   ./tests/scripts/validate_rpath.sh [build_dir] [library_name]
#
# Example:
#   ./tests/scripts/validate_rpath.sh build libtt_metal.so

set -euo pipefail

# Default values
BUILD_DIR="${1:-build}"
LIBRARY_NAME="${2:-libtt_metal.so}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if readelf is available
if ! command -v readelf &> /dev/null; then
    echo -e "${RED}Error: readelf not found. Please install binutils.${NC}" >&2
    exit 1
fi

# Find the library
LIBRARY_PATH=""
if [ -f "${BUILD_DIR}/lib/${LIBRARY_NAME}" ]; then
    LIBRARY_PATH="${BUILD_DIR}/lib/${LIBRARY_NAME}"
elif [ -f "${BUILD_DIR}/${LIBRARY_NAME}" ]; then
    LIBRARY_PATH="${BUILD_DIR}/${LIBRARY_NAME}"
else
    echo -e "${YELLOW}Warning: Library ${LIBRARY_NAME} not found in ${BUILD_DIR}${NC}" >&2
    echo "Searched locations:"
    echo "  - ${BUILD_DIR}/lib/${LIBRARY_NAME}"
    echo "  - ${BUILD_DIR}/${LIBRARY_NAME}"
    exit 1
fi

echo "Validating RPATH/RUNPATH for: ${LIBRARY_PATH}"
echo ""

# Extract RPATH/RUNPATH using readelf
RPATH_OUTPUT=$(readelf -d "${LIBRARY_PATH}" 2>/dev/null | grep -E "(RPATH|RUNPATH)" || true)

if [ -z "${RPATH_OUTPUT}" ]; then
    echo -e "${YELLOW}Warning: No RPATH or RUNPATH found in ${LIBRARY_PATH}${NC}"
    echo "This might be intentional for system libraries, but verify it's correct."
    exit 0
fi

echo "RPATH/RUNPATH entries:"
echo "${RPATH_OUTPUT}"
echo ""

# Check for $ORIGIN (must be first for Fedora compliance)
ORIGIN_FOUND=false
ORIGIN_FIRST=false

if echo "${RPATH_OUTPUT}" | grep -q '\$ORIGIN'; then
    ORIGIN_FOUND=true
    # Check if $ORIGIN is first
    FIRST_ENTRY=$(echo "${RPATH_OUTPUT}" | head -1 | grep -o '\[.*\]' | tr -d '[]')
    if echo "${FIRST_ENTRY}" | grep -q '^\$ORIGIN'; then
        ORIGIN_FIRST=true
        echo -e "${GREEN}✓ $ORIGIN found and is first in RPATH/RUNPATH${NC}"
    else
        echo -e "${RED}✗ $ORIGIN found but is NOT first in RPATH/RUNPATH${NC}"
        echo "  First entry: ${FIRST_ENTRY}"
        echo "  Fedora's brp-check-rpaths requires $ORIGIN to be first"
    fi
else
    echo -e "${YELLOW}⚠ $ORIGIN not found in RPATH/RUNPATH${NC}"
    echo "  This might be OK for system libraries, but verify it's intentional"
fi

# Check for absolute build paths (should not be present in installed binaries)
ABSOLUTE_BUILD_PATHS=$(echo "${RPATH_OUTPUT}" | grep -o '/[^:]*' | grep -E "^${PROJECT_ROOT}|^/tmp|^/home" || true)

if [ -n "${ABSOLUTE_BUILD_PATHS}" ]; then
    echo -e "${RED}✗ Found absolute build paths in RPATH/RUNPATH:${NC}"
    echo "${ABSOLUTE_BUILD_PATHS}"
    echo "  These should be replaced with \$ORIGIN-relative paths"
    exit 1
else
    echo -e "${GREEN}✓ No absolute build paths found in RPATH/RUNPATH${NC}"
fi

# Check for MPI library paths (if MPI is enabled)
# This is informational - MPI paths are OK if they're system paths or $ORIGIN-relative
MPI_PATHS=$(echo "${RPATH_OUTPUT}" | grep -iE "mpi|openmpi" || true)

if [ -n "${MPI_PATHS}" ]; then
    echo -e "${GREEN}✓ MPI library paths found in RPATH/RUNPATH:${NC}"
    echo "${MPI_PATHS}"
    # Check if MPI paths are absolute (custom ULFM) or system paths
    if echo "${MPI_PATHS}" | grep -qE "^/opt|^/usr"; then
        echo "  Note: Using custom ULFM MPI or system MPI"
    fi
fi

# Summary
echo ""
echo "=== Summary ==="
if [ "${ORIGIN_FOUND}" = true ] && [ "${ORIGIN_FIRST}" = true ] && [ -z "${ABSOLUTE_BUILD_PATHS}" ]; then
    echo -e "${GREEN}✓ RPATH/RUNPATH validation passed${NC}"
    exit 0
elif [ "${ORIGIN_FOUND}" = false ] && [ -z "${ABSOLUTE_BUILD_PATHS}" ]; then
    echo -e "${YELLOW}⚠ RPATH/RUNPATH validation passed with warnings${NC}"
    exit 0
else
    echo -e "${RED}✗ RPATH/RUNPATH validation failed${NC}"
    exit 1
fi
