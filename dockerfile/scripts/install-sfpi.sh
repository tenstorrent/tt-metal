#!/bin/bash
# Install SFPI compiler from tenstorrent/sfpi releases
# SFPI is a compiler for Tenstorrent's special functions processing instruction set
#
# This script downloads the pre-built SFPI .deb package and extracts it to INSTALL_DIR.
# The deb package installs files to /opt/tenstorrent/sfpi, which we extract to /install.
#
# IMPORTANT: Version and hash information come from tt_metal/sfpi-version and tt_metal/sfpi-info.sh
# Those files are the SINGLE SOURCE OF TRUTH for SFPI version information.
# Do NOT hardcode version/hash values in this script or in Dockerfile.tools.
set -euo pipefail


INSTALL_DIR="${INSTALL_DIR:-/install}"

# Source SFPI version information from the canonical source files
# These must be copied into the build context before this script runs
SFPI_INFO_SCRIPT="${SFPI_INFO_SCRIPT:-/tmp/sfpi-info.sh}"
SFPI_VERSION_FILE="${SFPI_VERSION_FILE:-/tmp/sfpi-version}"

if [[ ! -r "$SFPI_VERSION_FILE" ]]; then
    echo "ERROR: SFPI version file not found at $SFPI_VERSION_FILE" >&2
    exit 1
fi

if [[ ! -r "$SFPI_INFO_SCRIPT" ]]; then
    echo "ERROR: SFPI info script not found at $SFPI_INFO_SCRIPT" >&2
    exit 1
fi

# Source the version file to get sfpi_version, hashes, etc.
# shellcheck source=/dev/null
source "$SFPI_VERSION_FILE"

# For Docker tool image builds, we always target x86_64 debian .deb
SFPI_ARCH="x86_64"
SFPI_DIST="debian"
SFPI_PKG="deb"

# Get the hash for our target platform
SFPI_HASH_VAR="sfpi_${SFPI_ARCH}_${SFPI_DIST}_${SFPI_PKG}_hash"
SFPI_HASH="${!SFPI_HASH_VAR:-}"

if [[ -z "$SFPI_HASH" ]]; then
    echo "ERROR: No hash found for SFPI ${sfpi_version} ${SFPI_ARCH}/${SFPI_DIST}/${SFPI_PKG}" >&2
    echo "Available variables in sfpi-version:" >&2
    grep -E "^sfpi_" "$SFPI_VERSION_FILE" >&2 || true
    exit 1
fi

DOWNLOAD_URL="${sfpi_repo}/releases/download/${sfpi_version}/sfpi_${sfpi_version}_${SFPI_ARCH}_${SFPI_DIST}.${SFPI_PKG}"
TMPDIR="/tmp/sfpi-install"
DEB_FILE="${TMPDIR}/sfpi.deb"

echo "Installing SFPI ${sfpi_version} for ${SFPI_ARCH}/${SFPI_DIST}..."
echo "Download URL: ${DOWNLOAD_URL}"
echo "Expected ${sfpi_hashtype} hash: ${SFPI_HASH}"

# Create temp directory
mkdir -p "${TMPDIR}"

# Download (use curl if wget not available)
if command -v wget &> /dev/null; then
    wget -q -O "${DEB_FILE}" "${DOWNLOAD_URL}"
else
    curl -fsSL -o "${DEB_FILE}" "${DOWNLOAD_URL}"
fi

# Verify hash using the hash type from sfpi-version (typically sha256)
if ! echo "${SFPI_HASH}  ${DEB_FILE}" | "${sfpi_hashtype}sum" -c - ; then
    echo "[ERROR] ${sfpi_hashtype} checksum verification failed for ${DEB_FILE}. Aborting." >&2
    exit 1
fi

# Extract the deb package
# deb packages are ar archives containing:
#   - debian-binary (version info)
#   - control.tar.* (package metadata)
#   - data.tar.* (actual files)
cd "${TMPDIR}"
ar x "${DEB_FILE}"

# Find and extract the data tarball (could be .tar.gz, .tar.xz, .tar.zst, etc.)
DATA_TAR=$(ls data.tar.* 2>/dev/null | head -1)
if [[ -z "${DATA_TAR}" ]]; then
    echo "ERROR: Could not find data.tar.* in deb package" >&2
    exit 1
fi

# Create install directory structure
mkdir -p "${INSTALL_DIR}"

# Extract data tarball to a temp location first
mkdir -p "${TMPDIR}/extract"
tar -xf "${DATA_TAR}" -C "${TMPDIR}/extract"

# The deb installs to /opt/tenstorrent/sfpi - move that to INSTALL_DIR/opt/tenstorrent/sfpi
# This preserves the full path structure which the main Dockerfile expects
if [[ -d "${TMPDIR}/extract/opt" ]]; then
    cp -a "${TMPDIR}/extract/opt" "${INSTALL_DIR}/"
fi

# Also copy any other directories that might be in the package (e.g., /usr)
for dir in "${TMPDIR}/extract"/*; do
    if [[ -d "$dir" ]] && [[ "$(basename "$dir")" != "opt" ]]; then
        cp -a "$dir" "${INSTALL_DIR}/"
    fi
done

# Cleanup
rm -rf "${TMPDIR}"

# Verify installation
if [[ -d "${INSTALL_DIR}/opt/tenstorrent/sfpi" ]]; then
    echo "SFPI ${sfpi_version} installed successfully to ${INSTALL_DIR}/opt/tenstorrent/sfpi"
    ls -la "${INSTALL_DIR}/opt/tenstorrent/sfpi"
else
    echo "ERROR: SFPI installation directory not found" >&2
    exit 1
fi
