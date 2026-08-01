#!/bin/bash
# Install curl from the official release source tarball.
# Usage: CURL_VERSION=8.21.0 CURL_SHA256=... \
#        OPENSSL_VERSION=4.0.1 OPENSSL_SHA256=... \
#        INSTALL_PREFIX=/install ./install-curl.sh
#
# Built with a statically-linked OpenSSL (built from source below) + dynamic
# zlib/libpsl against whatever dev headers/libs are present in the builder
# stage, and installed to INSTALL_PREFIX/bin/curl. Placed ahead of apt's
# /usr/bin/curl on PATH in the final image so it's a drop-in replacement,
# not an addition.
#
# Why OpenSSL is statically linked while zlib/libpsl aren't: this tool image
# is built once (in a manylinux/AlmaLinux 9 builder, for ABI consistency with
# the rest of the from-source tools) and copied into BOTH the ubuntu-22.04
# and ubuntu-24.04 final images. OpenSSL uses versioned symbols per minor
# release (e.g. OPENSSL_3.2.0) and AlmaLinux 9 has shipped OpenSSL 3.2.x
# since 9.4, newer than either Ubuntu's 3.0.x - a curl dynamically linked
# against the builder's libssl.so crashed on both destinations with
# "version `OPENSSL_3.2.0' not found". zlib and libpsl don't have this
# versioned-symbol problem, so they stay dynamic as before.
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

OPENSSL_VERSION="${OPENSSL_VERSION:?OPENSSL_VERSION is required}"
# SHA256 for openssl-${OPENSSL_VERSION}.tar.gz from the official GitHub release
# (openssl/openssl), computed directly from a downloaded release archive.
OPENSSL_SHA256="${OPENSSL_SHA256:?OPENSSL_SHA256 is required}"

INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local}"
OPENSSL_URL="https://github.com/openssl/openssl/releases/download/openssl-${OPENSSL_VERSION}/openssl-${OPENSSL_VERSION}.tar.gz"
CURL_URL="https://curl.se/download/curl-${CURL_VERSION}.tar.gz"
OPENSSL_TMPDIR="/tmp/openssl-build"
OPENSSL_STATIC_PREFIX="/tmp/openssl-static"
CURL_TMPDIR="/tmp/curl-build"

fetch() {
    # fetch <url> <output-path>
    if command -v wget &> /dev/null; then
        wget -q -O "$2" "$1"
    else
        curl -fsSL -o "$2" "$1"
    fi
}

# ---------------------------------------------------------------------------
# Step 1: build a static OpenSSL. Curl links against this instead of the
# builder's system OpenSSL so the resulting binary has zero runtime
# dependency on whatever OpenSSL happens to be installed wherever it ends up.
# ---------------------------------------------------------------------------
echo "Building static OpenSSL ${OPENSSL_VERSION}..."
mkdir -p "${OPENSSL_TMPDIR}"
fetch "${OPENSSL_URL}" "${OPENSSL_TMPDIR}/openssl.tar.gz"

if ! echo "${OPENSSL_SHA256}  ${OPENSSL_TMPDIR}/openssl.tar.gz" | sha256sum -c - ; then
    echo "[ERROR] SHA256 checksum verification failed for openssl.tar.gz. Aborting." >&2
    exit 1
fi

tar -xf "${OPENSSL_TMPDIR}/openssl.tar.gz" -C "${OPENSSL_TMPDIR}" --strip-components=1
cd "${OPENSSL_TMPDIR}"
./Configure no-shared no-tests --prefix="${OPENSSL_STATIC_PREFIX}" --openssldir="${OPENSSL_STATIC_PREFIX}/ssl"
make -j"$(nproc)"
make install_sw install_ssldirs
cd /
rm -rf "${OPENSSL_TMPDIR}"

# ---------------------------------------------------------------------------
# Step 2: build curl against the static OpenSSL above.
# ---------------------------------------------------------------------------
echo "Installing curl ${CURL_VERSION}..."
mkdir -p "${CURL_TMPDIR}"
fetch "${CURL_URL}" "${CURL_TMPDIR}/curl.tar.gz"

if ! echo "${CURL_SHA256}  ${CURL_TMPDIR}/curl.tar.gz" | sha256sum -c - ; then
    echo "[ERROR] SHA256 checksum verification failed for curl.tar.gz. Aborting." >&2
    exit 1
fi

tar -xf "${CURL_TMPDIR}/curl.tar.gz" -C "${CURL_TMPDIR}" --strip-components=1

mkdir -p "${INSTALL_PREFIX}"

# zlib (compression) and libpsl (Public Suffix List - scopes cookie-jar
# cookies to their real registrable domain) are required here: configure
# fails outright if either isn't found, rather than silently building
# without it. This binary globally replaces /usr/bin/curl in the final image
# (see Dockerfile), so silently dropping libpsl would quietly weaken
# cookie-domain isolation for every curl consumer in the image, not just the
# Garage/SigV4 use case this tool exists for - keep the build failing loudly
# if the builder stage's libpsl-devel goes missing rather than reintroducing
# that regression silently.
# --with-openssl points at the static build from Step 1 (not the system
# copy); nghttp2 (HTTP/2) is the one true optional extra, picked up
# automatically if the builder stage installed its dev package.
#
# --with-ca-bundle/--with-ca-path are hardcoded to the Ubuntu/Debian
# location rather than left to autodetection: configure autodetects (and
# bakes in) whatever CA bundle path exists on the BUILD machine, which is
# manylinux/AlmaLinux 9's /etc/pki/tls/certs/ca-bundle.crt - a path that
# doesn't exist on either Ubuntu destination image, so every HTTPS request
# failed with "curl: (77) error adding trust anchors from file". Ubuntu's
# actual path (verified present on both 22.04 and 24.04) is
# /etc/ssl/certs/ca-certificates.crt.
cd "${CURL_TMPDIR}"
./configure \
    --prefix="${INSTALL_PREFIX}" \
    --with-openssl="${OPENSSL_STATIC_PREFIX}" \
    --with-ca-bundle=/etc/ssl/certs/ca-certificates.crt \
    --with-ca-path=/etc/ssl/certs
make -j"$(nproc)"
make install
cd /
rm -rf "${CURL_TMPDIR}" "${OPENSSL_STATIC_PREFIX}"

# Verify installation, and specifically that OpenSSL got linked in
# statically (no libssl.so.* dependency) - this is the property the whole
# static-OpenSSL build exists for, so catch a regression here at build time
# rather than downstream in some consumer image at runtime.
"${INSTALL_PREFIX}/bin/curl" --version
if ldd "${INSTALL_PREFIX}/bin/curl" 2>/dev/null | grep -qi 'libssl\.so'; then
    echo "[ERROR] curl is dynamically linked against libssl.so - OpenSSL should be statically linked. Aborting." >&2
    exit 1
fi
echo "curl ${CURL_VERSION} installed successfully (OpenSSL statically linked)"
