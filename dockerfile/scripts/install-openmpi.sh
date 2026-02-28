#!/bin/bash
# Build OpenMPI from source with ULFM (User Level Fault Mitigation) support.
# Used by Dockerfile.tools to produce a pre-built OpenMPI layer for manylinux.
# Output is installed under ${OMPI_PREFIX} (default /opt/openmpi-<tag>-ulfm).
set -euo pipefail

OMPI_TAG="${OMPI_TAG:-v5.0.7}"
INSTALL_DIR="${INSTALL_DIR:-/opt}"
OMPI_PREFIX="${INSTALL_DIR}/openmpi-${OMPI_TAG}-ulfm"

echo "Building OpenMPI ${OMPI_TAG} (from git) with ULFM to ${OMPI_PREFIX}..."

WORKDIR="/tmp/ompi-src"
rm -rf "${WORKDIR}"
mkdir -p "${WORKDIR}"

echo "Cloning OpenMPI ${OMPI_TAG} from GitHub..."
git clone --branch "${OMPI_TAG}" --depth 1 https://github.com/open-mpi/ompi.git "${WORKDIR}"
cd "${WORKDIR}"

# Run autogen.pl to generate configure script (required when building from git)
echo "Running autogen.pl..."
./autogen.pl

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

# Create mpirun-ulfm symlink for compatibility with scripts expecting this name
if [ -e "${OMPI_PREFIX}/bin/mpirun-ulfm" ]; then
    echo "[WARNING] mpirun-ulfm already exists at ${OMPI_PREFIX}/bin/mpirun-ulfm, skipping symlink creation"
elif [ ! -x "${OMPI_PREFIX}/bin/mpirun" ]; then
    echo "[ERROR] mpirun not found or not executable at ${OMPI_PREFIX}/bin/mpirun, cannot create mpirun-ulfm symlink" >&2
    exit 1
else
    cd "${OMPI_PREFIX}/bin" && ln -s mpirun mpirun-ulfm
fi

echo "OpenMPI ${OMPI_TAG} installed to ${OMPI_PREFIX}"
if [ -x "${OMPI_PREFIX}/bin/mpicc" ]; then
    "${OMPI_PREFIX}/bin/mpicc" --version || true
fi
