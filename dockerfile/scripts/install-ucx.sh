#!/bin/bash
# Build UCX from the release tarball, configured for Broadcom RoCE (generic verbs).
# Used by Dockerfile.tools to give the OpenMPI build something to point --with-ucx at.
# Output is installed under ${UCX_PREFIX} (default /opt/ucx-<version>).
#
# The transport set is deliberate: bnxt_re is a stock rdma-core provider, so RoCE runs
# through rc_verbs/ud_verbs. --with-verbs/--with-rc/--with-ud/--with-rdmacm are the four
# load-bearing options; without them UCX still builds but comes up with only self/tcp/sysv
# and osc/ucx would silently run at TCP speed. Everything mlx5-specific is switched OFF so
# configure cannot pick up Mellanox direct-verbs headers if they are present in the builder.
set -euo pipefail

UCX_VERSION="${UCX_VERSION:-1.22.0}"
INSTALL_DIR="${INSTALL_DIR:-/opt}"
UCX_PREFIX="${UCX_PREFIX:-${INSTALL_DIR}/ucx-${UCX_VERSION}}"
UCX_SHA256="${UCX_SHA256:-}"
UCX_JOBS="${UCX_JOBS:-4}"

TARBALL="ucx-${UCX_VERSION}.tar.gz"
URL="https://github.com/openucx/ucx/releases/download/v${UCX_VERSION}/${TARBALL}"

echo "Building UCX ${UCX_VERSION} to ${UCX_PREFIX}..."

WORKDIR="/tmp/ucx-src"
rm -rf "${WORKDIR}"
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

curl -fsSL -o "${TARBALL}" "${URL}"

if [ -n "${UCX_SHA256}" ]; then
    echo "${UCX_SHA256}  ${TARBALL}" | sha256sum -c -
    echo "==> Verified ${TARBALL} against UCX_SHA256"
else
    # Every other tool in Dockerfile.tools pins a hash (see ZSTD_SHA256). This one is
    # unpinned until someone fills in ARG UCX_SHA256, and says so rather than pretending.
    echo "[WARNING] UCX_SHA256 is empty -- ${TARBALL} was NOT checksum-verified." >&2
    echo "[WARNING] Pin it: sha256sum ${TARBALL}, then set ARG UCX_SHA256 in Dockerfile.tools." >&2
fi

tar xf "${TARBALL}"
cd "ucx-${UCX_VERSION}"

# The release tarball ships a pre-generated configure, so there is no autogen.sh step and
# none of the autoconf/automake/libtool build deps a git clone would need.
./configure \
    --prefix="${UCX_PREFIX}" \
    --enable-mt \
    --enable-cma \
    --with-verbs \
    --with-rc \
    --with-ud \
    --with-rdmacm \
    --without-mlx5 \
    --without-devx \
    --without-dc \
    --without-dm \
    --without-gga \
    --without-ib-hw-tm \
    --without-efa \
    --without-cuda \
    --without-rocm \
    --without-ze \
    --without-gaudi \
    --without-java \
    --without-go \
    --disable-doxygen-doc

make -j"${UCX_JOBS}"
make install -j"${UCX_JOBS}"
cd /
rm -rf "${WORKDIR}"

# GUARD: verbs is the whole point. A UCX without rc_verbs builds and installs cleanly and
# then runs everything over TCP, which looks like a working RDMA run at one tenth the
# bandwidth -- so fail the image build here rather than ship that.
if [ ! -e "${UCX_PREFIX}/lib/ucx/libuct_ib.so" ] && [ ! -e "${UCX_PREFIX}/lib/ucx/libuct_ib.a" ]; then
    echo "[ERROR] ${UCX_PREFIX}/lib/ucx/libuct_ib.* is missing: UCX built without verbs." >&2
    echo "[ERROR] libibverbs-devel / librdmacm-devel are required in the builder stage." >&2
    exit 1
fi
echo "==> Verified: uct_ib module present (verbs support built)"

echo "UCX ${UCX_VERSION} installed to ${UCX_PREFIX}"
if [ -x "${UCX_PREFIX}/bin/ucx_info" ]; then
    "${UCX_PREFIX}/bin/ucx_info" -v || true
fi
