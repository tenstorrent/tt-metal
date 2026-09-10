#!/bin/bash
# Build OpenMPI from source with ULFM **and a UCX RDMA one-sided path**.
#
# This is the RDMA variant of install-openmpi.sh, kept as a separate script so the mainline
# build is not carrying UCX plumbing it never uses. Used by Dockerfile.openmpi-rdma.
# Output is installed under ${OMPI_PREFIX} (default /opt/openmpi-<tag>-ulfm-rdma).
#
# UCX_PREFIX is REQUIRED here -- that is the whole point of this variant. Run install-ucx.sh
# first and point this at its prefix.
set -euo pipefail

OMPI_VERSION="${OMPI_VERSION:-v5.0.7}"
INSTALL_DIR="${INSTALL_DIR:-/opt}"
OMPI_PREFIX="${INSTALL_DIR}/openmpi-${OMPI_VERSION}-ulfm-rdma"
UCX_PREFIX="${UCX_PREFIX:?UCX_PREFIX is required for the RDMA build -- run install-ucx.sh first}"

echo "Building OpenMPI ${OMPI_VERSION} (from git) with ULFM + UCX(${UCX_PREFIX}) to ${OMPI_PREFIX}..."

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
    echo "WARNING: OMPI_COMMIT_SHA not set, cloning by tag ${OMPI_VERSION} (less secure)"
    git clone --branch "${OMPI_VERSION}" --depth 1 --recursive https://github.com/open-mpi/ompi.git "${WORKDIR}"
fi

cd "${WORKDIR}"

# Run autogen.pl to generate configure script (required when building from git)
echo "Running autogen.pl..."
./autogen.pl

# NOTE: OpenMPI configure options for ULFM fault tolerance and HPC environment compatibility.
# The --with-slurm=/opt/slurm option enables SLURM process management support for multi-host jobs.
# --disable-mca-dso and --disable-dlopen create a static build for portability.
# These settings were validated for the current manylinux-based environment.
# If issues arise with different MPI runtimes or cluster configurations, review with the scaleout team.
# See: https://github.com/open-mpi/ompi/blob/main/docs/features/ulfm.rst
#
# CFLAGS: force C17 (-std=gnu17). GCC 14 auto-selects C23 (gnu23) which promotes
# certain OpenMPI _Bool->volatile void* assignments to hard errors. GCC uses the
# last -std flag on the command line, so this overrides the gnu23 embedded in CC.
if [ ! -d "${UCX_PREFIX}/lib" ]; then
    echo "[ERROR] UCX_PREFIX=${UCX_PREFIX} has no lib/ -- run install-ucx.sh first." >&2
    exit 1
fi
export PKG_CONFIG_PATH="${UCX_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="${UCX_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# --with-ucx uses the =DIR form on purpose: with an explicit path configure HARD-FAILS if UCX
# is not there, instead of quietly probing system paths, finding nothing, and producing yet
# another MPI with no RDMA path. That silent outcome is what this variant exists to prevent.
#
# --enable-mca-no-build=pml-ucx keeps ob1 as the PML. ULFM's revoke/shrink machinery lives in
# the ob1+BTL path and pml/ucx does not support fault tolerance, so leaving pml/ucx buildable
# would let it win selection at run time and take ULFM with it. osc/ucx still provides the
# RDMA one-sided path under ob1, which is the combination this image is for.
CFLAGS="-std=gnu17" ./configure \
    --prefix="${OMPI_PREFIX}" \
    --with-ucx="${UCX_PREFIX}" \
    --with-ucx-libdir="${UCX_PREFIX}/lib" \
    --enable-mca-no-build=pml-ucx \
    --with-ft=ulfm \
    --enable-wrapper-rpath \
    --enable-mpirun-prefix-by-default \
    --disable-mca-dso \
    --disable-dlopen \
    --enable-static \
    --with-slurm=/opt/slurm \
    --without-munge \
    --with-pic

make -j"$(nproc)"
make install -j"$(nproc)"
cd /
rm -rf "${WORKDIR}"

# Create mpirun-ulfm wrapper for compatibility with scripts expecting this name.
# The wrapper calls mpirun with --with-ft ulfm and forwards all remaining arguments.
if [ -e "${OMPI_PREFIX}/bin/mpirun-ulfm" ]; then
    echo "[WARNING] mpirun-ulfm already exists at ${OMPI_PREFIX}/bin/mpirun-ulfm, skipping wrapper creation"
elif [ ! -x "${OMPI_PREFIX}/bin/mpirun" ]; then
    echo "[ERROR] mpirun not found or not executable at ${OMPI_PREFIX}/bin/mpirun, cannot create mpirun-ulfm wrapper" >&2
    exit 1
else
    cat > "${OMPI_PREFIX}/bin/mpirun-ulfm" <<'EOF'
#!/bin/bash
exec "$(dirname "$0")/mpirun" --with-ft ulfm "$@"
EOF
    chmod +x "${OMPI_PREFIX}/bin/mpirun-ulfm"
fi

# Guard: ensure no munge runtime dependency leaked into libmpi.so.
# munge-devel must not be present in the builder image; this catches it if it ever is.
if ldd "${OMPI_PREFIX}/lib/libmpi.so" 2>/dev/null | grep -q munge; then
    echo "[ERROR] libmpi.so links against libmunge — runtime containers will fail to load." >&2
    exit 1
fi
echo "==> Verified: libmpi.so has no munge runtime dependency"

# GUARD: the whole reason for --with-ucx is osc/ucx. Assert it is really there -- a configure
# that failed to find UCX still produces a perfectly working MPI, just one that cannot do RDMA
# one-sided, and that difference is invisible until a benchmark comes back ten times slow.
if ! "${OMPI_PREFIX}/bin/ompi_info" 2>/dev/null | grep -q "MCA osc: ucx"; then
    echo "[ERROR] osc/ucx is absent from the build despite --with-ucx=${UCX_PREFIX}." >&2
    "${OMPI_PREFIX}/bin/ompi_info" 2>/dev/null | grep -E "MCA (osc|pml|btl):" >&2 || true
    exit 1
fi
echo "==> Verified: osc/ucx built"

# GUARD: pml/ucx must NOT be present -- it would outbid ob1 and take ULFM with it.
if "${OMPI_PREFIX}/bin/ompi_info" 2>/dev/null | grep -q "MCA pml: ucx"; then
    echo "[ERROR] pml/ucx was built despite --enable-mca-no-build=pml-ucx; it would" >&2
    echo "[ERROR] outbid ob1 at run time and disable ULFM." >&2
    exit 1
fi
echo "==> Verified: pml/ucx excluded, ob1 keeps ULFM"

# GUARD: btl/uct is the other RDMA one-sided route (osc/rdma rides it). Its absence is not
# fatal -- osc/ucx alone is a working configuration -- but it is worth saying out loud, because
# picking --mca osc rdma later with no uct silently leaves you on tcp.
if "${OMPI_PREFIX}/bin/ompi_info" 2>/dev/null | grep -q "MCA btl: uct"; then
    echo "==> Verified: btl/uct built (osc/rdma has an RDMA BTL)"
else
    echo "[WARNING] btl/uct was NOT built: '--mca osc rdma --mca osc_rdma_btls uct' will find" >&2
    echo "[WARNING] no RDMA BTL and fall back to tcp without saying so." >&2
fi

# GUARD: SLURM launch lives in PRRTE under OpenMPI 5, NOT in ompi_info. Checking the wrong tool
# reports zero components on a build that has them.
if [ -x "${OMPI_PREFIX}/bin/prte_info" ]; then
    if ! "${OMPI_PREFIX}/bin/prte_info" 2>/dev/null | grep -qE "ras: *slurm"; then
        echo "[ERROR] PRRTE has no ras/slurm: mpirun will not read a SLURM allocation and" >&2
        echo "[ERROR] --map-by ppr:N:node will map only the local node." >&2
        exit 1
    fi
    echo "==> Verified: PRRTE ras/slurm present"
fi

echo "OpenMPI ${OMPI_VERSION} installed to ${OMPI_PREFIX}"
if [ -x "${OMPI_PREFIX}/bin/mpicc" ]; then
    "${OMPI_PREFIX}/bin/mpicc" --version || true
fi
