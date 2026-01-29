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
echo "${CCACHE_SHA256}  ${TMPFILE}" | sha256sum -c -

# Extract to install directory
# The tarball contains ccache-X.Y.Z-linux-x86_64/{bin/ccache, ...}
mkdir -p "${INSTALL_DIR}/bin"
tar -xf "${TMPFILE}" -C "${INSTALL_DIR}" --strip-components=1

# Cleanup
rm -f "${TMPFILE}"

# Verify installation
"${INSTALL_DIR}/bin/ccache" --version
echo "ccache ${CCACHE_VERSION} installed successfully"
