#!/bin/bash
# Install ClangBuildAnalyzer from source
set -euo pipefail

CBA_VERSION="${CBA_VERSION:-1.6.0}"
# SHA256 for ClangBuildAnalyzer source tarball v1.6.0
# Verified by downloading and computing hash with compute-hashes.sh
CBA_SHA256="${CBA_SHA256:-868a8d34ecb9b65da4e5874342062a12c081ce4385c7ddd6ce7d557a0c5c292d}"

INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local}"
DOWNLOAD_URL="https://github.com/aras-p/ClangBuildAnalyzer/archive/refs/tags/v${CBA_VERSION}.tar.gz"
TMPDIR="/tmp/cba"

echo "Installing ClangBuildAnalyzer ${CBA_VERSION}..."

# Create temp directory
mkdir -p "${TMPDIR}"

# Download (use curl if wget not available)
if command -v wget &> /dev/null; then
    wget -q -O "${TMPDIR}/cba.tar.gz" "${DOWNLOAD_URL}"
else
    curl -fsSL -o "${TMPDIR}/cba.tar.gz" "${DOWNLOAD_URL}"
fi

# Verify hash
if ! echo "${CBA_SHA256}  ${TMPDIR}/cba.tar.gz" | sha256sum -c - ; then
    echo "[ERROR] SHA256 checksum verification failed for cba.tar.gz. Aborting." >&2
    exit 1
fi

# Extract
tar -xzf "${TMPDIR}/cba.tar.gz" -C "${TMPDIR}" --strip-components=1

# Create install prefix directory
mkdir -p "${INSTALL_PREFIX}"

# Build with CMake
cmake -S "${TMPDIR}" -B "${TMPDIR}/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}"
cmake --build "${TMPDIR}/build" -j"$(nproc)"
cmake --install "${TMPDIR}/build"

# Cleanup
rm -rf "${TMPDIR}"

# Verify installation
"${INSTALL_PREFIX}/bin/ClangBuildAnalyzer" --version || echo "ClangBuildAnalyzer installed (no --version flag)"
echo "ClangBuildAnalyzer ${CBA_VERSION} installed successfully"
