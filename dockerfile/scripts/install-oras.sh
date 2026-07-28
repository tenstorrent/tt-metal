#!/bin/bash
# Install oras (OCI Registry As Storage CLI) from upstream binary release
# oras-project/oras is not available in apt, so we download from GitHub releases
set -euo pipefail

ORAS_VERSION="${ORAS_VERSION:-1.3.3}"
# SHA256 for oras_1.3.3_linux_amd64.tar.gz
# Verified from GitHub release (matches the pin previously carried in
# .github/scripts/manual-docker-bake.sh)
ORAS_SHA256="${ORAS_SHA256:-9ce999f8d2de03fc03968b29d743077a58783e545e5eaa53917ca177352d0e59}"

INSTALL_DIR="${INSTALL_DIR:-/usr/local}"
DOWNLOAD_URL="https://github.com/oras-project/oras/releases/download/v${ORAS_VERSION}/oras_${ORAS_VERSION}_linux_amd64.tar.gz"
TMPFILE="/tmp/oras.tar.gz"

echo "Installing oras ${ORAS_VERSION}..."

# Download (use curl if wget not available)
if command -v wget &> /dev/null; then
    wget -q -O "${TMPFILE}" "${DOWNLOAD_URL}"
else
    curl -fsSL -o "${TMPFILE}" "${DOWNLOAD_URL}"
fi

# Verify hash
if ! echo "${ORAS_SHA256}  ${TMPFILE}" | sha256sum -c - ; then
    echo "[ERROR] SHA256 checksum verification failed for ${TMPFILE}. Aborting." >&2
    exit 1
fi

# The tarball contains the oras binary at its root alongside LICENSE/README.
TMPDIR_EXTRACT=$(mktemp -d)
tar -xzf "${TMPFILE}" -C "${TMPDIR_EXTRACT}" oras

mkdir -p "${INSTALL_DIR}/bin"
install -m 0755 "${TMPDIR_EXTRACT}/oras" "${INSTALL_DIR}/bin/oras"

# Cleanup
rm -rf "${TMPDIR_EXTRACT}"
rm -f "${TMPFILE}"

# Verify installation (skip if binary can't run, e.g., glibc binary on musl/Alpine)
if "${INSTALL_DIR}/bin/oras" version 2>/dev/null; then
    echo "oras ${ORAS_VERSION} installed and verified successfully"
else
    echo "oras ${ORAS_VERSION} installed (verification skipped - binary may require glibc)"
fi
