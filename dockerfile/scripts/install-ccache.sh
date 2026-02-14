#!/bin/bash
# Install ccache from upstream binary release
# Apt's version for 20.04 predates remote_storage support
set -euo pipefail

CCACHE_VERSION="${CCACHE_VERSION:-4.10.2}"
# SHA256 for ccache-4.10.2-linux-x86_64.tar.xz
# Verified by downloading and computing hash with compute-hashes.sh
CCACHE_SHA256="${CCACHE_SHA256:-80cab87bd510eca796467aee8e663c398239e0df1c4800a0b5dff11dca0b4f18}"

INSTALL_DIR="${INSTALL_DIR:-/usr/local}"
DOWNLOAD_URL="https://github.com/ccache/ccache/releases/download/v${CCACHE_VERSION}/ccache-${CCACHE_VERSION}-linux-x86_64.tar.xz"
TMPFILE="/tmp/ccache.tar.xz"

echo "Installing ccache ${CCACHE_VERSION}..."

# Download (use curl if wget not available)
if command -v wget &> /dev/null; then
    wget -q -O "${TMPFILE}" "${DOWNLOAD_URL}"
else
    curl -fsSL -o "${TMPFILE}" "${DOWNLOAD_URL}"
fi

# Verify hash
if ! echo "${CCACHE_SHA256}  ${TMPFILE}" | sha256sum -c - ; then
    echo "[ERROR] SHA256 checksum verification failed for ${TMPFILE}. Aborting." >&2
    exit 1
fi

# Extract to install directory
# The tarball contains ccache-X.Y.Z-linux-x86_64/ccache (binary at root, no bin/ subdir)
mkdir -p "${INSTALL_DIR}/bin"
tar -xf "${TMPFILE}" -C "${INSTALL_DIR}" --strip-components=1

# Move binary to bin/ for standard layout
mv "${INSTALL_DIR}/ccache" "${INSTALL_DIR}/bin/ccache"

# Cleanup
rm -f "${TMPFILE}"

# Verify installation (skip if binary can't run, e.g., glibc binary on musl/Alpine)
if "${INSTALL_DIR}/bin/ccache" --version 2>/dev/null; then
    echo "ccache ${CCACHE_VERSION} installed and verified successfully"
else
    echo "ccache ${CCACHE_VERSION} installed (verification skipped - binary may require glibc)"
fi
