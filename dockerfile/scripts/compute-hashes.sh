#!/bin/bash
# Compute SHA256 hashes for tool tarballs. Use when updating versions in Dockerfile.tools.
#
# IMPORTANT: Keep version defaults in sync with dockerfile/Dockerfile.tools.
# After updating Dockerfile.tools ARGs, run this script and update the SHA256 ARGs.
set -euo pipefail

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

# Use shasum on macOS, sha256sum on Linux
if command -v sha256sum &> /dev/null; then
    SHA_CMD="sha256sum"
elif command -v shasum &> /dev/null; then
    SHA_CMD="shasum -a 256"
else
    echo "ERROR: Neither sha256sum nor shasum found"
    exit 1
fi

echo "Computing SHA256 hashes for tool downloads..."
echo "=============================================="
echo ""

# ccache
CCACHE_VERSION="${CCACHE_VERSION:-4.10.2}"
echo "Downloading ccache ${CCACHE_VERSION}..."
curl -fsSL -o "$TMPDIR/ccache.tar.xz" \
    "https://github.com/ccache/ccache/releases/download/v${CCACHE_VERSION}/ccache-${CCACHE_VERSION}-linux-x86_64.tar.xz"
echo "CCACHE_SHA256=$($SHA_CMD "$TMPDIR/ccache.tar.xz" | cut -d' ' -f1)"
echo ""

# mold
MOLD_VERSION="${MOLD_VERSION:-2.40.4}"
echo "Downloading mold ${MOLD_VERSION}..."
curl -fsSL -o "$TMPDIR/mold.tar.gz" \
    "https://github.com/rui314/mold/releases/download/v${MOLD_VERSION}/mold-${MOLD_VERSION}-x86_64-linux.tar.gz"
echo "MOLD_SHA256=$($SHA_CMD "$TMPDIR/mold.tar.gz" | cut -d' ' -f1)"
echo ""

# doxygen
DOXYGEN_VERSION="${DOXYGEN_VERSION:-1.16.1}"
echo "Downloading doxygen ${DOXYGEN_VERSION}..."
curl -fsSL -o "$TMPDIR/doxygen.tar.gz" \
    "https://www.doxygen.nl/files/doxygen-${DOXYGEN_VERSION}.linux.bin.tar.gz"
echo "DOXYGEN_SHA256=$($SHA_CMD "$TMPDIR/doxygen.tar.gz" | cut -d' ' -f1)"
echo ""

# ClangBuildAnalyzer
CBA_VERSION="${CBA_VERSION:-1.6.0}"
echo "Downloading ClangBuildAnalyzer ${CBA_VERSION}..."
curl -fsSL -o "$TMPDIR/cba.tar.gz" \
    "https://github.com/aras-p/ClangBuildAnalyzer/archive/refs/tags/v${CBA_VERSION}.tar.gz"
echo "CBA_SHA256=$($SHA_CMD "$TMPDIR/cba.tar.gz" | cut -d' ' -f1)"
echo ""

# GDB
GDB_VERSION="${GDB_VERSION:-14.2}"
echo "Downloading GDB ${GDB_VERSION}..."
curl -fsSL -o "$TMPDIR/gdb.tar.xz" \
    "https://mirror.csclub.uwaterloo.ca/gnu/gdb/gdb-${GDB_VERSION}.tar.xz"
echo "GDB_SHA256=$($SHA_CMD "$TMPDIR/gdb.tar.xz" | cut -d' ' -f1)"
echo ""

# cmake
CMAKE_VERSION="${CMAKE_VERSION:-4.2.3}"
echo "Downloading cmake ${CMAKE_VERSION}..."
curl -fsSL -o "$TMPDIR/cmake.tar.gz" \
    "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz"
echo "CMAKE_SHA256=$($SHA_CMD "$TMPDIR/cmake.tar.gz" | cut -d' ' -f1)"
echo ""

# yq
YQ_VERSION="${YQ_VERSION:-v4.44.6}"
echo "Downloading yq ${YQ_VERSION}..."
curl -fsSL -o "$TMPDIR/yq_linux_amd64" \
    "https://github.com/mikefarah/yq/releases/download/${YQ_VERSION}/yq_linux_amd64"
echo "YQ_SHA256=$($SHA_CMD "$TMPDIR/yq_linux_amd64" | cut -d' ' -f1)"
echo ""

# OpenMPI
OMPI_TAG="${OMPI_TAG:-v5.0.7}"
OMPI_VERSION="${OMPI_TAG#v}"
OMPI_SERIES="v${OMPI_VERSION%.*}"
echo "Downloading OpenMPI ${OMPI_TAG}..."
curl -fsSL -o "$TMPDIR/openmpi.tar.bz2" \
    "https://download.open-mpi.org/release/open-mpi/${OMPI_SERIES}/openmpi-${OMPI_VERSION}.tar.bz2"
echo "OMPI_SHA256=$($SHA_CMD "$TMPDIR/openmpi.tar.bz2" | cut -d' ' -f1)"
echo ""

# SFPI - Note: version comes from tt_metal/sfpi-version, not hardcoded here
echo "SFPI: Version and hash come from tt_metal/sfpi-version (single source of truth)"
echo "       Run: grep sfpi_x86_64_debian_deb_hash tt_metal/sfpi-version"
echo ""

echo "=============================================="
echo "Copy these values to the ARG declarations in Dockerfile.tools"
echo ""
echo "Note: SFPI version/hash are NOT in Dockerfile.tools."
echo "      They come from tt_metal/sfpi-version (single source of truth)."
