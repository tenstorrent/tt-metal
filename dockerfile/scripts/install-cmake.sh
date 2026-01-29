#!/bin/bash
# Install cmake from official tarball
# Usage: CMAKE_VERSION=4.2.3 CMAKE_SHA256=... INSTALL_DIR=/install ./install-cmake.sh

set -e

CMAKE_VERSION=${CMAKE_VERSION:?CMAKE_VERSION is required}
CMAKE_SHA256=${CMAKE_SHA256:?CMAKE_SHA256 is required}
INSTALL_DIR=${INSTALL_DIR:-/usr/local}

TARBALL_URL="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz"
TARBALL_FILE="cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz"

echo "[INFO] Downloading cmake ${CMAKE_VERSION}..."
wget -q --show-progress "${TARBALL_URL}" -O "/tmp/${TARBALL_FILE}"

echo "[INFO] Verifying SHA256 checksum..."
echo "${CMAKE_SHA256}  /tmp/${TARBALL_FILE}" | sha256sum -c -

echo "[INFO] Extracting cmake..."
mkdir -p /tmp/cmake-extract
tar -xzf "/tmp/${TARBALL_FILE}" -C /tmp/cmake-extract --strip-components=1

echo "[INFO] Installing cmake to ${INSTALL_DIR}..."
mkdir -p "${INSTALL_DIR}"
cp -r /tmp/cmake-extract/* "${INSTALL_DIR}/"

echo "[INFO] Cleaning up..."
rm -rf /tmp/cmake-extract "/tmp/${TARBALL_FILE}"

echo "[INFO] cmake ${CMAKE_VERSION} installed successfully"
"${INSTALL_DIR}/bin/cmake" --version
