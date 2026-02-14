#!/bin/bash
# Build OpenMPI from source with ULFM (User Level Fault Mitigation) support.
# Used by Dockerfile.tools to produce a pre-built OpenMPI layer for manylinux.
# Output is installed under ${OMPI_PREFIX} (default /opt/openmpi-<tag>-ulfm).
#
# We build from the official release tarball (not git) so that library SONAMEs
# are stable (e.g. libopen-pal.so.80.0.5). Building from git would embed the
# commit hash in the library name (e.g. libopen-pal-e07e4432.so.80.0.5), which
# would break the manylinux wheel at test time when the loader looks for that
# exact name.
set -euo pipefail

OMPI_TAG="${OMPI_TAG:-v5.0.7}"
# SHA256 for openmpi-5.0.7.tar.bz2 from https://download.open-mpi.org/release/open-mpi/v5.0/
# Update via: OMPI_TAG=v5.0.7 dockerfile/scripts/compute-hashes.sh
OMPI_SHA256="${OMPI_SHA256:-119f2009936a403334d0df3c0d74d5595a32d99497f9b1d41e90019fee2fc2dd}"
INSTALL_DIR="${INSTALL_DIR:-/opt}"
OMPI_PREFIX="${INSTALL_DIR}/openmpi-${OMPI_TAG}-ulfm"

# Derive version and URL from tag (e.g. v5.0.7 -> 5.0.7, release path v5.0)
OMPI_VERSION="${OMPI_TAG#v}"
OMPI_SERIES="v${OMPI_VERSION%.*}"
OMPI_TARBALL="openmpi-${OMPI_VERSION}.tar.bz2"
OMPI_URL="https://download.open-mpi.org/release/open-mpi/${OMPI_SERIES}/${OMPI_TARBALL}"

echo "Building OpenMPI ${OMPI_TAG} (from tarball) with ULFM to ${OMPI_PREFIX}..."

WORKDIR="/tmp/ompi-src"
rm -rf "${WORKDIR}"
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"
wget -q -O "${OMPI_TARBALL}" "${OMPI_URL}"
if ! echo "${OMPI_SHA256}  ${OMPI_TARBALL}" | sha256sum -c - ; then
    echo "[ERROR] SHA256 checksum verification failed for ${OMPI_TARBALL}. Aborting." >&2
    exit 1
fi
tar xf "${OMPI_TARBALL}"
cd "openmpi-${OMPI_VERSION}"

# TODO: Evaluate with scaleout team
./configure \
    --prefix="${OMPI_PREFIX}" \
    --with-ft=ulfm \
    --enable-wrapper-rpath \
    --enable-mpirun-prefix-by-default \
    --disable-mca-dso \
    --disable-dlopen \
    --enable-static \
    --with-slurm \
    --with-pic

make -j"$(nproc)"
make install
cd /
rm -rf "${WORKDIR}"

echo "OpenMPI ${OMPI_TAG} installed to ${OMPI_PREFIX}"
if [ -x "${OMPI_PREFIX}/bin/mpicc" ]; then
    "${OMPI_PREFIX}/bin/mpicc" --version || true
fi
