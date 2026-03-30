#!/bin/bash
# Build OpenMPI from source with ULFM (User Level Fault Mitigation) support.
# Used by Dockerfile.tools to produce a pre-built OpenMPI layer for manylinux.
# Output is installed under ${OMPI_PREFIX} (default /opt/openmpi-<tag>-ulfm).
set -euo pipefail

OMPI_TAG="${OMPI_TAG:-v5.0.10}"
INSTALL_DIR="${INSTALL_DIR:-/opt}"
OMPI_PREFIX="${INSTALL_DIR}/openmpi-${OMPI_TAG}-ulfm"

echo "Building OpenMPI ${OMPI_TAG} (from git) with ULFM to ${OMPI_PREFIX}..."

WORKDIR="/tmp/ompi-src"
rm -rf "${WORKDIR}"
mkdir -p "${WORKDIR}"

if [ -n "${OMPI_COMMIT_SHA:-}" ]; then
    echo "Fetching OpenMPI commit ${OMPI_COMMIT_SHA} from GitHub..."
    git init "${WORKDIR}"
    git -C "${WORKDIR}" remote add origin https://github.com/open-mpi/ompi.git
    git -C "${WORKDIR}" fetch --depth 1 origin "${OMPI_COMMIT_SHA}"
    git -C "${WORKDIR}" checkout FETCH_HEAD
    git -C "${WORKDIR}" submodule update --init --recursive
else
    echo "WARNING: OMPI_COMMIT_SHA not set, cloning by tag ${OMPI_TAG} (less secure)"
    git clone --branch "${OMPI_TAG}" --depth 1 --recursive https://github.com/open-mpi/ompi.git "${WORKDIR}"
fi

cd "${WORKDIR}"

# Run autogen.pl to generate configure script (required when building from git)
echo "Running autogen.pl..."
./autogen.pl

# NOTE: OpenMPI configure options for ULFM fault tolerance and HPC environment compatibility.
# The --with-slurm option enables SLURM process management support for multi-host jobs.
# --disable-mca-dso and --disable-dlopen create a static build for portability.
# These settings were validated for the current manylinux-based environment.
# If issues arise with different MPI runtimes or cluster configurations, review with the scaleout team.
# See: https://github.com/open-mpi/ompi/blob/main/docs/features/ulfm.rst
./configure \
    --prefix="${OMPI_PREFIX}" \
    --with-ft=ulfm \
    --enable-wrapper-rpath \
    --enable-prte-prefix-by-default \
    --disable-mca-dso \
    --disable-dlopen \
    --enable-static \
    --with-slurm \
    --with-pic

make -j"$(nproc)"
make install -j"$(nproc)"
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
