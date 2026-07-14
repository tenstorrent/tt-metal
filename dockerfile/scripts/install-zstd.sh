#!/bin/bash
# Install Zstandard (zstd) from the official release source tarball.
# Usage: ZSTD_VERSION=1.5.7 ZSTD_SHA256=... INSTALL_PREFIX=/install ./install-zstd.sh
set -euo pipefail

ZSTD_VERSION="${ZSTD_VERSION:?ZSTD_VERSION is required}"
# SHA256 for zstd-${ZSTD_VERSION}.tar.gz from the GitHub release.
# Verified against the upstream zstd-${ZSTD_VERSION}.tar.gz.sha256 sidecar.
ZSTD_SHA256="${ZSTD_SHA256:?ZSTD_SHA256 is required}"

INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local}"
DOWNLOAD_URL="https://github.com/facebook/zstd/releases/download/v${ZSTD_VERSION}/zstd-${ZSTD_VERSION}.tar.gz"
TMPDIR="/tmp/zstd-build"

echo "Installing zstd ${ZSTD_VERSION}..."

# Create temp directory
mkdir -p "${TMPDIR}"

# Download (use curl if wget not available)
if command -v wget &> /dev/null; then
    wget -q -O "${TMPDIR}/zstd.tar.gz" "${DOWNLOAD_URL}"
else
    curl -fsSL -o "${TMPDIR}/zstd.tar.gz" "${DOWNLOAD_URL}"
fi

# Verify hash
if ! echo "${ZSTD_SHA256}  ${TMPDIR}/zstd.tar.gz" | sha256sum -c - ; then
    echo "[ERROR] SHA256 checksum verification failed for zstd.tar.gz. Aborting." >&2
    exit 1
fi

# Extract
tar -xf "${TMPDIR}/zstd.tar.gz" -C "${TMPDIR}" --strip-components=1

# Create install prefix directory
mkdir -p "${INSTALL_PREFIX}"

# Build and install (CLI binary + libzstd + headers + man pages)
cd "${TMPDIR}"
make -j"$(nproc)"
make install PREFIX="${INSTALL_PREFIX}"

# Cleanup
rm -rf "${TMPDIR}"

# Verify installation
"${INSTALL_PREFIX}/bin/zstd" --version
echo "zstd ${ZSTD_VERSION} installed successfully"
