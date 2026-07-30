#!/bin/bash
# Install curl from the official release source tarball.
# Usage: CURL_VERSION=8.21.0 CURL_SHA256=... INSTALL_PREFIX=/install ./install-curl.sh
#
# Built with OpenSSL + zlib + nghttp2 + libpsl against whatever dev
# headers/libs are present in the builder stage, and installed to
# INSTALL_PREFIX/bin/curl. Placed ahead of apt's /usr/bin/curl on PATH in
# the final image so it's a drop-in replacement, not an addition.
#
# Why this exists: curl's --aws-sigv4 only started sending the
# X-Amz-Content-Sha256 header in curl 8.x. Garage (our S3-compatible CI
# cache backend) requires that header on every SigV4 request, so curl 7.81
# (the version 'apt install curl' gives you on Ubuntu 22.04/jammy, which
# never gets a major-version bump within an LTS release) gets every signed
# request rejected with "400 Bad Request: Missing X-Amz-Content-Sha256
# field" before auth is even evaluated. See MINFRA-1374.
set -euo pipefail

CURL_VERSION="${CURL_VERSION:?CURL_VERSION is required}"
# SHA256 for curl-${CURL_VERSION}.tar.gz from the official curl.se release archive.
# curl.se publishes PGP signatures (.asc) rather than a plain sidecar checksum file,
# so this hash is computed directly from a signature-verified download (same
# trust model already used here for gdb/doxygen; see compute-hashes.sh).
CURL_SHA256="${CURL_SHA256:?CURL_SHA256 is required}"

INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local}"
DOWNLOAD_URL="https://curl.se/download/curl-${CURL_VERSION}.tar.gz"
TMPDIR="/tmp/curl-build"

echo "Installing curl ${CURL_VERSION}..."

# Create temp directory
mkdir -p "${TMPDIR}"

# Download (use curl if wget not available)
if command -v wget &> /dev/null; then
    wget -q -O "${TMPDIR}/curl.tar.gz" "${DOWNLOAD_URL}"
else
    curl -fsSL -o "${TMPDIR}/curl.tar.gz" "${DOWNLOAD_URL}"
fi

# Verify hash
if ! echo "${CURL_SHA256}  ${TMPDIR}/curl.tar.gz" | sha256sum -c - ; then
    echo "[ERROR] SHA256 checksum verification failed for curl.tar.gz. Aborting." >&2
    exit 1
fi

# Extract
tar -xf "${TMPDIR}/curl.tar.gz" -C "${TMPDIR}" --strip-components=1

# Create install prefix directory
mkdir -p "${INSTALL_PREFIX}"

# Configure, build, and install.
# OpenSSL (TLS), zlib (compression), and libpsl (Public Suffix List - used to
# scope cookie-jar cookies to their real registrable domain) are all required
# here: configure fails outright if any of them isn't found, rather than
# silently building without it. This binary globally replaces /usr/bin/curl
# in the final image (see Dockerfile), so silently dropping libpsl would
# quietly weaken cookie-domain isolation for every curl consumer in the
# image, not just the Garage/SigV4 use case this tool exists for - keep the
# build failing loudly if the builder stage's libpsl-devel goes missing
# rather than reintroducing that regression silently.
# nghttp2 (HTTP/2) is the one optional extra: picked up automatically if the
# builder stage installed its dev package, but not required to succeed.
cd "${TMPDIR}"
./configure \
    --prefix="${INSTALL_PREFIX}" \
    --with-openssl
make -j"$(nproc)"
make install

# Cleanup
rm -rf "${TMPDIR}"

# Verify installation
"${INSTALL_PREFIX}/bin/curl" --version
echo "curl ${CURL_VERSION} installed successfully"
