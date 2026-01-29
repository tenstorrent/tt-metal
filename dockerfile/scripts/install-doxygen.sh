#!/bin/bash
# Install doxygen from upstream binary release
set -euo pipefail

DOXYGEN_VERSION="${DOXYGEN_VERSION:-1.9.6}"
# SHA256 for doxygen-1.9.6.linux.bin.tar.gz
# Verified by downloading and computing hash with compute-hashes.sh
DOXYGEN_SHA256="${DOXYGEN_SHA256:-8354583f86416586d35397c8ee7e719f5aa5804140af83cf7ba39a8c5076bdb8}"

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
echo "${DOXYGEN_SHA256}  ${TMPDIR}/doxygen.tar.gz" | sha256sum -c -

# Extract
tar -xzf "${TMPDIR}/doxygen.tar.gz" -C "${TMPDIR}" --strip-components=1

# Create install prefix directory
mkdir -p "${INSTALL_PREFIX}/bin"

# Build and install (doxygen binary release uses DESTDIR for installation root)
make -C "${TMPDIR}" -j"$(nproc)"
# Use DESTDIR to redirect installation - files go to $DESTDIR/usr/local/...
make -C "${TMPDIR}" install DESTDIR="${INSTALL_PREFIX}"

# Move from DESTDIR structure to flat structure for easier COPY in Dockerfile
# /staging/usr/local/bin/doxygen -> /staging/bin/doxygen
if [ -d "${INSTALL_PREFIX}/usr/local/bin" ]; then
    mv "${INSTALL_PREFIX}/usr/local/bin/"* "${INSTALL_PREFIX}/bin/" 2>/dev/null || true
    rm -rf "${INSTALL_PREFIX}/usr"
fi

# Cleanup
rm -rf "${TMPDIR}"

# Verify installation
"${INSTALL_PREFIX}/bin/doxygen" --version
echo "doxygen ${DOXYGEN_VERSION} installed successfully"
