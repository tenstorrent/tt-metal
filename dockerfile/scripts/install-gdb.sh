#!/bin/bash
# Install GDB from source (for Ubuntu versions without recent GDB)
set -euo pipefail

GDB_VERSION="${GDB_VERSION:-14.2}"
# SHA256 for gdb-14.2.tar.xz from GNU FTP
# Verified from: https://lists.gnu.org/archive/html/info-gnu/2024-03/msg00000.html
GDB_SHA256="${GDB_SHA256:-2d4dd8061d8ded12b6c63f55e45344881e8226105f4d2a9b234040efa5ce7772}"

INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local}"
DOWNLOAD_URL="https://mirror.csclub.uwaterloo.ca/gnu/gdb/gdb-${GDB_VERSION}.tar.xz"
TMPDIR="/tmp/gdb-build"

echo "Installing GDB ${GDB_VERSION}..."

# Create temp directory
mkdir -p "${TMPDIR}"

# Download (use curl if wget not available)
if command -v wget &> /dev/null; then
    wget -q -O "${TMPDIR}/gdb.tar.xz" "${DOWNLOAD_URL}"
else
    curl -fsSL -o "${TMPDIR}/gdb.tar.xz" "${DOWNLOAD_URL}"
fi

# Verify hash
if ! echo "${GDB_SHA256}  ${TMPDIR}/gdb.tar.xz" | sha256sum -c - ; then
    echo "[ERROR] SHA256 checksum verification failed for gdb.tar.xz. Aborting." >&2
    exit 1
fi

# Extract
tar -xf "${TMPDIR}/gdb.tar.xz" -C "${TMPDIR}" --strip-components=1

# Create install prefix directory
mkdir -p "${INSTALL_PREFIX}"

# Configure and build
cd "${TMPDIR}"
./configure --prefix="${INSTALL_PREFIX}"
make -j"$(nproc)"
make install

# Cleanup
rm -rf "${TMPDIR}"

# Verify installation
"${INSTALL_PREFIX}/bin/gdb" --version
echo "GDB ${GDB_VERSION} installed successfully"
