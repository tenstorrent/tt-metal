#!/bin/bash
# Install ccache from upstream binary release
# Apt's version for 20.04 predates remote_storage support
set -euo pipefail

CCACHE_VERSION="${CCACHE_VERSION:-4.13.2}"
# SHA256 for ccache-4.13.2-linux-x86_64-glibc.tar.xz
# Note: starting v4.11, upstream renamed the tarball to include -glibc suffix
# Verified by downloading and computing hash with compute-hashes.sh
CCACHE_SHA256="${CCACHE_SHA256:-e9e2bcec3cd816ba58ff1c331fdeaa3465760683eeec9f462c31b909e2651e35}"

INSTALL_DIR="${INSTALL_DIR:-/usr/local}"
DOWNLOAD_URL="https://github.com/ccache/ccache/releases/download/v${CCACHE_VERSION}/ccache-${CCACHE_VERSION}-linux-x86_64-glibc.tar.xz"
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
