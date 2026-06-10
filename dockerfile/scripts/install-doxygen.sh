#!/bin/bash
# Install doxygen from upstream binary release
set -euo pipefail

DOXYGEN_VERSION="${DOXYGEN_VERSION:-1.16.1}"
# SHA256 for doxygen-1.16.1.linux.bin.tar.gz
# Verified by downloading and computing hash with compute-hashes.sh
DOXYGEN_SHA256="${DOXYGEN_SHA256:-a56f885d37e3aae08a99f638d17bbb381224c03a878d9e2dda4f9fa4baf1d8bd}"

INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local}"
DOWNLOAD_URL="https://www.doxygen.nl/files/doxygen-${DOXYGEN_VERSION}.linux.bin.tar.gz"
TMPDIR="/tmp/doxygen"

echo "Installing doxygen ${DOXYGEN_VERSION}..."

# Create temp directory
mkdir -p "${TMPDIR}"

# Download (use curl if wget not available)
if command -v wget &> /dev/null; then
    wget -q -O "${TMPDIR}/doxygen.tar.gz" "${DOWNLOAD_URL}"
else
    curl -fsSL -o "${TMPDIR}/doxygen.tar.gz" "${DOWNLOAD_URL}"
fi

# Verify hash
if ! echo "${DOXYGEN_SHA256}  ${TMPDIR}/doxygen.tar.gz" | sha256sum -c - ; then
    echo "[ERROR] SHA256 checksum verification failed for doxygen.tar.gz. Aborting." >&2
    exit 1
fi

# Extract
tar -xzf "${TMPDIR}/doxygen.tar.gz" -C "${TMPDIR}" --strip-components=1

# Create install directory
mkdir -p "${INSTALL_PREFIX}/bin"

# The doxygen binary release Makefile has hardcoded paths, so copy directly
# The binary is at bin/doxygen after extraction
cp "${TMPDIR}/bin/doxygen" "${INSTALL_PREFIX}/bin/doxygen"
chmod 755 "${INSTALL_PREFIX}/bin/doxygen"

# Cleanup
rm -rf "${TMPDIR}"

# Verify installation
"${INSTALL_PREFIX}/bin/doxygen" --version
echo "doxygen ${DOXYGEN_VERSION} installed successfully"
