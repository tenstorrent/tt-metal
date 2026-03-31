#!/bin/bash
# Build and install Slurm headers + libslurm from source.
# Provides the development files needed for OpenMPI's --with-slurm configure option.
# Only headers and the client library are installed — no daemons or plugins.
set -euo pipefail

SLURM_VERSION="25.11.4"
SLURM_TARBALL="slurm-${SLURM_VERSION}.tar.bz2"
SLURM_URL="https://github.com/SchedMD/slurm/releases/download/slurm-25-11-4-1/${SLURM_TARBALL}"
SLURM_SHA256="237f515adcb37b99ff6cb3e1fa81691a5f7188e94c3b6b495858b4eaecf6531d"
SLURM_PREFIX="${SLURM_PREFIX:-/opt/slurm}"

WORKDIR="/tmp/slurm-build"
rm -rf "${WORKDIR}"
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

echo "==> Downloading Slurm ${SLURM_VERSION}..."
curl -fSL -o "${SLURM_TARBALL}" "${SLURM_URL}"

echo "==> Verifying SHA256..."
echo "${SLURM_SHA256}  ${SLURM_TARBALL}" | sha256sum -c -

echo "==> Extracting..."
tar xf "${SLURM_TARBALL}"
cd "slurm-${SLURM_VERSION}"

echo "==> Configuring (headers + libs only)..."
./configure \
    --prefix="${SLURM_PREFIX}" \
    --disable-debug \
    --disable-x11 \
    --disable-sview \
    --disable-slurmrestd \
    --without-rpath \
    --without-readline

echo "==> Building libslurm..."
# Build only the API library — we don't need daemons, plugins, or CLI tools.
# This produces libslurm.{a,so} which is all OpenMPI links against.
make -j"$(nproc)" -C src/api

echo "==> Installing headers and libraries to ${SLURM_PREFIX}..."
mkdir -p "${SLURM_PREFIX}/include" "${SLURM_PREFIX}/lib"

# Install headers (slurm/*.h)
cp -a slurm "${SLURM_PREFIX}/include/"

# Install libraries
cp -a src/api/.libs/libslurm.so* "${SLURM_PREFIX}/lib/"
cp -a src/api/.libs/libslurm.a   "${SLURM_PREFIX}/lib/" 2>/dev/null || true

# Create pkgconfig for discoverability
mkdir -p "${SLURM_PREFIX}/lib/pkgconfig"
cat > "${SLURM_PREFIX}/lib/pkgconfig/slurm.pc" <<PCEOF
prefix=${SLURM_PREFIX}
libdir=\${prefix}/lib
includedir=\${prefix}/include

Name: slurm
Description: Slurm client library
Version: ${SLURM_VERSION}
Libs: -L\${libdir} -lslurm
Cflags: -I\${includedir}
PCEOF

echo "==> Cleaning up build directory..."
cd /
rm -rf "${WORKDIR}"

echo "==> Slurm ${SLURM_VERSION} headers + libs installed to ${SLURM_PREFIX}"
ls -la "${SLURM_PREFIX}/include/slurm/" "${SLURM_PREFIX}/lib/"
