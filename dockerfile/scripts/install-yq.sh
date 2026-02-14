#!/bin/bash
# Install yq (YAML processor) from upstream binary release
# mikefarah/yq is not available in apt, so we download from GitHub releases
set -euo pipefail

YQ_VERSION="${YQ_VERSION:-v4.44.6}"
# SHA256 for yq_linux_amd64 v4.44.6
# Verified from GitHub release
YQ_SHA256="${YQ_SHA256:-0c2b24e645b57d8e7c0566d18643a6d4f5580feeea3878127354a46f2a1e4598}"

INSTALL_DIR="${INSTALL_DIR:-/usr/local}"
DOWNLOAD_URL="https://github.com/mikefarah/yq/releases/download/${YQ_VERSION}/yq_linux_amd64"
TMPFILE="/tmp/yq"

echo "Installing yq ${YQ_VERSION}..."

# Download (use curl if wget not available)
if command -v wget &> /dev/null; then
    wget -q -O "${TMPFILE}" "${DOWNLOAD_URL}"
else
    curl -fsSL -o "${TMPFILE}" "${DOWNLOAD_URL}"
fi

# Verify hash
if ! echo "${YQ_SHA256}  ${TMPFILE}" | sha256sum -c - ; then
    echo "[ERROR] SHA256 checksum verification failed for ${TMPFILE}. Aborting." >&2
    exit 1
fi

# Install to bin directory
mkdir -p "${INSTALL_DIR}/bin"
cp "${TMPFILE}" "${INSTALL_DIR}/bin/yq"
chmod +x "${INSTALL_DIR}/bin/yq"

# Cleanup
rm -f "${TMPFILE}"

# Verify installation (skip if binary can't run, e.g., glibc binary on musl/Alpine)
if "${INSTALL_DIR}/bin/yq" --version 2>/dev/null; then
    echo "yq ${YQ_VERSION} installed and verified successfully"
else
    echo "yq ${YQ_VERSION} installed (verification skipped - binary may require glibc)"
fi
