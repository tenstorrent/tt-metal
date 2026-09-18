#!/bin/bash
# Install mold linker from upstream binary release for faster linking
set -euo pipefail

MOLD_VERSION="${MOLD_VERSION:-2.42.0}"
# SHA256 for mold-2.42.0-x86_64-linux.tar.gz
# Verified from GitHub release
MOLD_SHA256="${MOLD_SHA256:-f5ed2f6e31d1ada4f07fe766fe0de7a73104d1c5cdc59086fcecc16a43720b6d}"

INSTALL_DIR="${INSTALL_DIR:-/usr/local}"
DOWNLOAD_URL="https://github.com/rui314/mold/releases/download/v${MOLD_VERSION}/mold-${MOLD_VERSION}-x86_64-linux.tar.gz"
TMPFILE="/tmp/mold.tar.gz"

echo "Installing mold ${MOLD_VERSION}..."

# Download (use curl if wget not available)
if command -v wget &> /dev/null; then
    wget -q -O "${TMPFILE}" "${DOWNLOAD_URL}"
else
    curl -fsSL -o "${TMPFILE}" "${DOWNLOAD_URL}"
fi

# Verify hash
if ! echo "${MOLD_SHA256}  ${TMPFILE}" | sha256sum -c - ; then
    echo "[ERROR] SHA256 checksum verification failed for ${TMPFILE}. Aborting." >&2
    exit 1
fi

# Extract to install directory
# The tarball contains mold-X.Y.Z-x86_64-linux/{bin/mold, lib/mold/...}
mkdir -p "${INSTALL_DIR}"
tar -xzf "${TMPFILE}" -C "${INSTALL_DIR}" --strip-components=1

# Cleanup
rm -f "${TMPFILE}"

# Verify installation (skip if binary can't run, e.g., glibc binary on musl/Alpine)
if "${INSTALL_DIR}/bin/mold" --version 2>/dev/null; then
    echo "mold ${MOLD_VERSION} installed and verified successfully"
else
    echo "mold ${MOLD_VERSION} installed (verification skipped - binary may require glibc)"
fi
