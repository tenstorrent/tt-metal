#!/bin/bash
# Build OpenMPI from source with ULFM (User Level Fault Mitigation) support.
# Used by Dockerfile.tools to produce a pre-built OpenMPI layer for manylinux.
# Output is installed under ${INSTALL_DIR}/opt/openmpi-${OMPI_TAG}-ulfm/
set -euo pipefail

OMPI_TAG="${OMPI_TAG:-v5.0.7}"
INSTALL_DIR="${INSTALL_DIR:-/install}"
OMPI_PREFIX="${INSTALL_DIR}/opt/openmpi-${OMPI_TAG}-ulfm"

echo "Building OpenMPI ${OMPI_TAG} with ULFM to ${OMPI_PREFIX}..."

WORKDIR="/tmp/ompi-src"
rm -rf "${WORKDIR}"
git clone --branch "${OMPI_TAG}" --depth 1 https://github.com/open-mpi/ompi.git "${WORKDIR}"
cd "${WORKDIR}"
git submodule update --init --recursive
./autogen.pl
./configure \
    --prefix="${OMPI_PREFIX}" \
    --with-ft=ulfm \
    --enable-wrapper-rpath \
    --enable-mpirun-prefix-by-default \
    --disable-mca-dso \
    --disable-dlopen
make -j"$(nproc)"
make install
cd /
rm -rf "${WORKDIR}"

echo "OpenMPI ${OMPI_TAG} installed to ${OMPI_PREFIX}"
if [ -x "${OMPI_PREFIX}/bin/mpicc" ]; then
    "${OMPI_PREFIX}/bin/mpicc" --version || true
fi
