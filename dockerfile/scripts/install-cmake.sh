#!/bin/bash
# Install cmake from official tarball
# Usage: CMAKE_VERSION=4.2.3 CMAKE_SHA256=... INSTALL_DIR=/install ./install-cmake.sh

set -euo pipefail

CMAKE_VERSION=${CMAKE_VERSION:?CMAKE_VERSION is required}
CMAKE_SHA256=${CMAKE_SHA256:?CMAKE_SHA256 is required}
INSTALL_DIR=${INSTALL_DIR:-/usr/local}

TARBALL_URL="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz"
TMPFILE="/tmp/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz"

echo "Installing cmake ${CMAKE_VERSION}..."

# Download (use curl if wget not available)
if command -v wget &> /dev/null; then
    wget -q --show-progress "${TARBALL_URL}" -O "${TMPFILE}"
else
    curl -fsSL -o "${TMPFILE}" "${TARBALL_URL}"
fi

# Verify hash
if ! echo "${CMAKE_SHA256}  ${TMPFILE}" | sha256sum -c - ; then
    echo "[ERROR] SHA256 checksum verification failed for ${TMPFILE}. Aborting." >&2
    exit 1
fi

# Extract to install directory
# The tarball contains cmake-X.Y.Z-linux-x86_64/{bin, doc, man, share}
mkdir -p "${INSTALL_DIR}"
tar -xzf "${TMPFILE}" -C "${INSTALL_DIR}" --strip-components=1

# move doc and man into share to match expected locations
# Use POSIX-compatible mv (no -t flag) for Alpine BusyBox compatibility
mv "${INSTALL_DIR}"/man "${INSTALL_DIR}"/doc "${INSTALL_DIR}"/share/

# Cleanup
rm -f "${TMPFILE}"

# Verify installation (skip if binary can't run, e.g., glibc binary on musl/Alpine)
if "${INSTALL_DIR}/bin/cmake" --version 2>/dev/null; then
    echo "cmake ${CMAKE_VERSION} installed and verified successfully"
else
    echo "cmake ${CMAKE_VERSION} installed (verification skipped - binary may require glibc)"
fi
