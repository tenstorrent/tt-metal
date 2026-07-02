#!/bin/bash
# Install ccache from upstream binary release
# Apt's version for 20.04 predates remote_storage support
set -euo pipefail

CCACHE_VERSION="${CCACHE_VERSION:-4.13.6}"
# SHA256 for ccache-4.13.6-linux-x86_64-glibc.tar.xz
# Note: starting v4.11, upstream renamed the tarball to include -glibc suffix
# Verified by downloading and computing hash with compute-hashes.sh
CCACHE_SHA256="${CCACHE_SHA256:-508b2a1217dc6e04a23e967c7b95a0fb45d8a7e16fde9e180919698f2e2be060}"

# Remote storage helper for S3-backed ccache (crsh: custom storage protocol).
# ccache derives the helper name (ccache-storage-s3) from the s3:// URL scheme
# and looks for it in its libexec_dirs (baked in as /usr/local/libexec below).
STORAGE_HELPER_VERSION="${STORAGE_HELPER_VERSION:-0.1.7}"
# SHA256 for ccache-storage-s3-go-0.1.7-linux-amd64.tar.gz
# Verified by downloading from GitHub releases and running sha256sum.
STORAGE_HELPER_SHA256="${STORAGE_HELPER_SHA256:-3ac52edbcbda895c5514f13eac39e6d9a9e2550234b331c8e14050a94b79854c}"

INSTALL_DIR="${INSTALL_DIR:-/usr/local}"
DOWNLOAD_URL="https://github.com/ccache/ccache/releases/download/v${CCACHE_VERSION}/ccache-${CCACHE_VERSION}-linux-x86_64-glibc.tar.xz"
TMPFILE="/tmp/ccache.tar.xz"

echo "Installing ccache ${CCACHE_VERSION}..."

# Download (use curl if wget not available)
if command -v wget &> /dev/null; then
    wget -q -O "${TMPFILE}" "${DOWNLOAD_URL}"
else
    curl -fsSL -o "${TMPFILE}" "${DOWNLOAD_URL}"
fi

# Verify hash
if ! echo "${CCACHE_SHA256}  ${TMPFILE}" | sha256sum -c - ; then
    echo "[ERROR] SHA256 checksum verification failed for ${TMPFILE}. Aborting." >&2
    exit 1
fi

# Extract to temp dir and run upstream install.sh
# install.sh uses patch-binary.py (requires python3) to bake the correct
# libexecdir and sysconfdir paths into the binary at install time.
# We pass the final on-disk paths (/usr/local/libexec, /etc) so
# ccache can locate its storage helper after the COPY --from stage lands
# the /install tree at /usr/local/ in the final image.
TMPDIR_EXTRACT=$(mktemp -d)
tar -xf "${TMPFILE}" -C "${TMPDIR_EXTRACT}" --strip-components=1

mkdir -p "${INSTALL_DIR}/bin"
"${TMPDIR_EXTRACT}/install.sh" \
    --prefix="${INSTALL_DIR}" \
    --libexecdir=/usr/local/libexec \
    --sysconfdir=/etc

# The storage helper is the ccache binary itself, invoked as a background
# daemon for async remote storage.  ccache looks for it in libexec_dirs
# (baked in above as /usr/local/libexec).  Place a copy there so it is
# present after the COPY --from stage maps /install/ → /usr/local/.
mkdir -p "${INSTALL_DIR}/libexec"
cp "${INSTALL_DIR}/bin/ccache" "${INSTALL_DIR}/libexec/ccache"

# Cleanup
rm -rf "${TMPDIR_EXTRACT}"
rm -f "${TMPFILE}"

# ---------------------------------------------------------------------------
# Install the S3 remote storage helper (blozano-tt/ccache-storage-s3-go).
# When CCACHE_REMOTE_STORAGE uses crsh: with an s3:// url, ccache derives the
# helper binary name (ccache-storage-s3) from the s3 scheme and looks it up in
# its libexec_dirs (/usr/local/libexec, baked into the ccache binary above).
# ---------------------------------------------------------------------------
echo "Installing ccache-storage-s3 helper ${STORAGE_HELPER_VERSION}..."

STORAGE_HELPER_URL="https://github.com/blozano-tt/ccache-storage-s3-go/releases/download/v${STORAGE_HELPER_VERSION}/ccache-storage-s3-go-${STORAGE_HELPER_VERSION}-linux-amd64.tar.gz"
STORAGE_HELPER_TMPFILE="/tmp/ccache-storage-s3.tar.gz"

# Download (use curl if wget not available)
if command -v wget &> /dev/null; then
    wget -q -O "${STORAGE_HELPER_TMPFILE}" "${STORAGE_HELPER_URL}"
else
    curl -fsSL -o "${STORAGE_HELPER_TMPFILE}" "${STORAGE_HELPER_URL}"
fi

# Verify hash
if ! echo "${STORAGE_HELPER_SHA256}  ${STORAGE_HELPER_TMPFILE}" | sha256sum -c - ; then
    echo "[ERROR] SHA256 checksum verification failed for ${STORAGE_HELPER_TMPFILE}. Aborting." >&2
    exit 1
fi

# The tarball contains a single binary. Extract it and install into libexec
# under the name ccache derives from the s3:// scheme: ccache-storage-s3.
STORAGE_HELPER_EXTRACT=$(mktemp -d)
tar -xf "${STORAGE_HELPER_TMPFILE}" -C "${STORAGE_HELPER_EXTRACT}"
STORAGE_HELPER_BIN=$(find "${STORAGE_HELPER_EXTRACT}" -type f -name 'ccache-storage-s3*' | head -n 1)
if [ -z "${STORAGE_HELPER_BIN}" ]; then
    echo "[ERROR] Could not find ccache-storage-s3 binary in ${STORAGE_HELPER_TMPFILE}. Aborting." >&2
    exit 1
fi
install -m 0755 "${STORAGE_HELPER_BIN}" "${INSTALL_DIR}/libexec/ccache-storage-s3"

# Cleanup
rm -rf "${STORAGE_HELPER_EXTRACT}"
rm -f "${STORAGE_HELPER_TMPFILE}"

# Verify installation (skip if binary can't run, e.g., glibc binary on musl/Alpine)
if "${INSTALL_DIR}/bin/ccache" --version 2>/dev/null; then
    echo "ccache ${CCACHE_VERSION} installed and verified successfully"
else
    echo "ccache ${CCACHE_VERSION} installed (verification skipped - binary may require glibc)"
fi
