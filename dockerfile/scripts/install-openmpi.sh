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

# Fix C23 build failure with GCC 14 in part_persist.h.
# OpenMPI's configure auto-detects GCC 14 on manylinux_2_34 and sets CC="gcc -std=gnu23".
# In C23, `false` is `_Bool`, and `req->req_ompi.req_complete = false` in part_persist.h
# assigns `_Bool` to `volatile void *` — a constraint violation (hard error, not a warning).
# The field req_complete is `volatile void *` and the rest of the codebase uses
# REQUEST_PENDING ((void *)0L) and REQUEST_COMPLETED ((void *)1L) for assignments.
# This one occurrence uses `false` then immediately overwrites via OPAL_ATOMIC_SWAP_PTR
# with REQUEST_PENDING — so the `false` assignment is both type-incorrect and redundant.
# Fix: replace `= false` with `= REQUEST_PENDING` to match the codebase convention.
echo "Fixing C23 part_persist type error (req_complete = false -> REQUEST_PENDING)..."
sed -i 's/req->req_ompi\.req_complete = false;/req->req_ompi.req_complete = REQUEST_PENDING;/' \
    ompi/mca/part/persist/part_persist.h

# Run autogen.pl to generate configure script (required when building from git)
echo "Running autogen.pl..."
./autogen.pl

# NOTE: OpenMPI configure options for ULFM fault tolerance and HPC environment compatibility.
# The --with-slurm=/opt/slurm option enables SLURM process management support for multi-host jobs.
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
    --with-slurm=/opt/slurm \
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
