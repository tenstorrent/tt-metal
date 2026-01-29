#!/bin/bash
# Utility script to compute SHA256 hashes for tool tarballs
# Run this to verify/update hashes in the Dockerfile
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
DOXYGEN_VERSION="${DOXYGEN_VERSION:-1.9.6}"
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

echo "=============================================="
echo "Copy these values to the ARG declarations in Dockerfile"
