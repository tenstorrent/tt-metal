#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Package pre-built fabric test binaries into a self-contained redistributable tarball.
# Run this from the tt-metal repo root after a successful build.
#
# Usage:
#   ./tests/tt_metal/tt_fabric/utils/package_fabric_tests.sh [--build-dir BUILD_DIR] [--output OUTPUT.tar.zst]
#
# Requirements:
#   - A completed tt-metal build (default: build/)
#   - patchelf (for RPATH fixup)
#   - tar + zstd (for compression)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"
OUTPUT=""
BUNDLE_SFPI=0

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --build-dir DIR     Path to build directory (default: build/)"
    echo "  --output FILE       Output tarball path (default: tt-fabric-tests-<date>.tar.zst)"
    echo "  --bundle-sfpi       Include SFPI toolchain in the package (adds ~500MB)"
    echo "  -h, --help          Show this help"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --output)    OUTPUT="$2"; shift 2 ;;
        --bundle-sfpi) BUNDLE_SFPI=1; shift ;;
        -h|--help)   usage ;;
        *)           echo "Unknown option: $1"; usage ;;
    esac
done

if [[ -z "$OUTPUT" ]]; then
    OUTPUT="${REPO_ROOT}/tt-fabric-tests-$(date +%Y%m%d).tar.zst"
fi

STAGING_DIR="$(mktemp -d)"
PKG_ROOT="${STAGING_DIR}/tt-fabric-tests"
trap 'rm -rf "${STAGING_DIR}"' EXIT

echo "==> Packaging fabric tests from: ${REPO_ROOT}"
echo "    Build directory: ${BUILD_DIR}"
echo "    Output: ${OUTPUT}"

# --- Validate prerequisites ---
BINARY="${BUILD_DIR}/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric"
if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: test_tt_fabric binary not found at ${BINARY}"
    echo "       Build with: cmake --build build -- test_tt_fabric"
    exit 1
fi

MUX_BINARY="${BUILD_DIR}/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric_mux_bandwidth"

# --- Create package layout ---
mkdir -p "${PKG_ROOT}"/{bin,lib}

echo "==> Copying test binaries..."
cp "${BINARY}" "${PKG_ROOT}/bin/"
if [[ -x "$MUX_BINARY" ]]; then
    cp "${MUX_BINARY}" "${PKG_ROOT}/bin/"
fi

# --- Harvest shared libraries via ldd ---
echo "==> Harvesting shared library dependencies..."
harvest_libs() {
    local binary="$1"
    ldd "$binary" 2>/dev/null | while read -r line; do
        # Match lines like: libfoo.so => /path/to/libfoo.so (0x...)
        if [[ "$line" =~ =\>\ (/[^ ]+) ]]; then
            local lib_path="${BASH_REMATCH[1]}"
            # Skip system libs that are always present (libc, libm, ld-linux, libpthread, etc.)
            local lib_name
            lib_name="$(basename "$lib_path")"
            case "$lib_name" in
                libc.so*|libm.so*|libdl.so*|librt.so*|libpthread.so*|ld-linux*|libgcc_s*|libstdc++*)
                    continue ;;
            esac
            echo "$lib_path"
        fi
    done
}

# Collect unique libs from all binaries
declare -A SEEN_LIBS
for bin in "${PKG_ROOT}"/bin/*; do
    while IFS= read -r lib; do
        if [[ -n "$lib" && -z "${SEEN_LIBS[$lib]:-}" ]]; then
            SEEN_LIBS["$lib"]=1
            cp -L "$lib" "${PKG_ROOT}/lib/"
        fi
    done < <(harvest_libs "$bin")
done
echo "    Collected ${#SEEN_LIBS[@]} shared libraries"

# --- Fix RPATH on binaries to use $ORIGIN/../lib ---
echo "==> Fixing RPATH on binaries..."
if command -v patchelf &>/dev/null; then
    for bin in "${PKG_ROOT}"/bin/*; do
        patchelf --set-rpath '$ORIGIN/../lib' "$bin"
    done
else
    echo "WARNING: patchelf not found. Binaries will rely on LD_LIBRARY_PATH."
    echo "         Install with: apt install patchelf"
fi

# --- Copy JIT kernel sources ---
echo "==> Copying JIT kernel sources..."

# Test-specific kernels
KERNEL_DIR="tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels"
mkdir -p "${PKG_ROOT}/${KERNEL_DIR}"
cp -a "${REPO_ROOT}/${KERNEL_DIR}/"*.{cpp,hpp} "${PKG_ROOT}/${KERNEL_DIR}/" 2>/dev/null || true

# Fabric mux kernel
MUX_KERNEL_DIR="tt_metal/fabric/impl/kernels"
mkdir -p "${PKG_ROOT}/${MUX_KERNEL_DIR}"
cp "${REPO_ROOT}/${MUX_KERNEL_DIR}/tt_fabric_mux.cpp" "${PKG_ROOT}/${MUX_KERNEL_DIR}/"

# Common test kernel utility headers
COMMON_KERNEL_DIR="tests/tt_metal/tt_metal/perf_microbenchmark/common"
mkdir -p "${PKG_ROOT}/${COMMON_KERNEL_DIR}"
cp "${REPO_ROOT}/${COMMON_KERNEL_DIR}/kernel_utils.hpp" "${PKG_ROOT}/${COMMON_KERNEL_DIR}/" 2>/dev/null || true

# --- Copy JIT include directories ---
echo "==> Copying JIT include/header trees..."

# JIT include directories from tt_metal/jit_build/build.cpp plus transitive deps.
# Use tt_metal/fabric (not just fabric/hw) because kernel sources #include
# top-level fabric headers like "fabric/fabric_edm_packet_header.hpp" via -I{root}/tt_metal.
JIT_INCLUDE_DIRS=(
    "tt_metal/hw/inc"
    "tt_metal/hw/ckernels"
    "tt_metal/hw/firmware/src"
    "tt_metal/hw/toolchain"
    "tt_metal/hw/lib"
    "tt_metal/tt-llk"
    "tt_metal/hostdevcommon"
    "tt_metal/api"
    "tt_metal/fabric"
    "tt_metal/tools"
    "tt_metal/impl/dispatch/kernels"
    "tt_metal/kernels"
    "ttnn/cpp"
    "runtime/hw/toolchain"
    "runtime/hw/lib"
)

for dir in "${JIT_INCLUDE_DIRS[@]}"; do
    if [[ -d "${REPO_ROOT}/${dir}" ]]; then
        mkdir -p "${PKG_ROOT}/${dir}"
        rsync -a --include='*/' \
            --include='*.hpp' --include='*.h' --include='*.cpp' \
            --include='*.cc' --include='*.c' --include='*.S' \
            --include='*.inc' --include='*.ld' --include='*.o' \
            --include='*.def' \
            --exclude='*' \
            "${REPO_ROOT}/${dir}/" "${PKG_ROOT}/${dir}/"
    fi
done

# Top-level tt_metal headers needed for JIT includes (the -I{root}tt_metal path)
for header_dir in "tt_metal/impl" "tt_metal/include"; do
    if [[ -d "${REPO_ROOT}/${header_dir}" ]]; then
        mkdir -p "${PKG_ROOT}/${header_dir}"
        rsync -a --include='*/' --include='*.hpp' --include='*.h' --exclude='*' \
            "${REPO_ROOT}/${header_dir}/" "${PKG_ROOT}/${header_dir}/"
    fi
done

# --- Copy runtime data files ---
echo "==> Copying runtime data files..."

# SoC descriptors
mkdir -p "${PKG_ROOT}/tt_metal/soc_descriptors"
cp "${REPO_ROOT}/tt_metal/soc_descriptors/"*.yaml "${PKG_ROOT}/tt_metal/soc_descriptors/"

# Core descriptors
mkdir -p "${PKG_ROOT}/tt_metal/core_descriptors"
cp "${REPO_ROOT}/tt_metal/core_descriptors/"*.yaml "${PKG_ROOT}/tt_metal/core_descriptors/"

# Mesh graph descriptors (cluster topology)
mkdir -p "${PKG_ROOT}/tt_metal/fabric/mesh_graph_descriptors"
cp "${REPO_ROOT}/tt_metal/fabric/mesh_graph_descriptors/"*.textproto "${PKG_ROOT}/tt_metal/fabric/mesh_graph_descriptors/"

# Pre-compiled firmware (if present)
if [[ -d "${REPO_ROOT}/tt_metal/pre-compiled" ]]; then
    echo "    Including pre-compiled firmware..."
    mkdir -p "${PKG_ROOT}/tt_metal/pre-compiled"
    cp -a "${REPO_ROOT}/tt_metal/pre-compiled/"* "${PKG_ROOT}/tt_metal/pre-compiled/"
fi

# Note: dispatch kernels (tt_metal/impl/dispatch/kernels), fabric EDM kernels
# (tt_metal/fabric/impl/kernels), and base metal kernels (tt_metal/kernels)
# are already copied by the JIT_INCLUDE_DIRS rsync loop above.

# --- Copy test YAML configs ---
echo "==> Copying test YAML configurations..."
YAML_DIR="tests/tt_metal/tt_metal/perf_microbenchmark/routing"
mkdir -p "${PKG_ROOT}/${YAML_DIR}"
cp "${REPO_ROOT}/${YAML_DIR}/"*.yaml "${PKG_ROOT}/${YAML_DIR}/"

# --- Copy multihost rank binding configs ---
echo "==> Copying multihost rank binding configurations..."
RANK_BINDING_DIR="tests/tt_metal/distributed/config"
mkdir -p "${PKG_ROOT}/${RANK_BINDING_DIR}"
cp "${REPO_ROOT}/${RANK_BINDING_DIR}/"*_rank_bindings.yaml "${PKG_ROOT}/${RANK_BINDING_DIR}/"

# --- Copy tt-run launcher (for multihost MPI tests) ---
echo "==> Copying tt-run launcher..."
TTRUN_DIR="ttnn/ttnn/distributed"
mkdir -p "${PKG_ROOT}/${TTRUN_DIR}"
cp "${REPO_ROOT}/${TTRUN_DIR}/ttrun.py" "${PKG_ROOT}/${TTRUN_DIR}/"

# --- Optionally bundle SFPI ---
if [[ "$BUNDLE_SFPI" -eq 1 ]]; then
    echo "==> Bundling SFPI toolchain..."
    SFPI_SRC=""
    if [[ -d "${REPO_ROOT}/runtime/sfpi" ]]; then
        SFPI_SRC="${REPO_ROOT}/runtime/sfpi"
    elif [[ -d "/opt/tenstorrent/sfpi" ]]; then
        SFPI_SRC="/opt/tenstorrent/sfpi"
    fi
    if [[ -n "$SFPI_SRC" ]]; then
        mkdir -p "${PKG_ROOT}/runtime/sfpi"
        cp -a "${SFPI_SRC}/"* "${PKG_ROOT}/runtime/sfpi/"
        echo "    SFPI bundled from: ${SFPI_SRC}"
    else
        echo "WARNING: SFPI toolchain not found. Target must have /opt/tenstorrent/sfpi."
    fi
else
    echo "    SFPI not bundled (target must have /opt/tenstorrent/sfpi)."
    echo "    Use --bundle-sfpi to include it."
fi

# --- Create wrapper script ---
echo "==> Creating wrapper script..."
cat > "${PKG_ROOT}/run_test.sh" << 'WRAPPER_EOF'
#!/usr/bin/env bash
# Wrapper script for running pre-built fabric tests.
# Sets up the environment so the binary can find its runtime dependencies.
#
# Usage:
#   ./run_test.sh --test_config tests/.../test_fabric_sanity_common.yaml [OTHER_ARGS...]
#   ./run_test.sh --binary test_tt_fabric_mux_bandwidth [ARGS...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

BINARY_NAME="test_tt_fabric"
ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --binary)
            BINARY_NAME="$2"
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

BINARY="${SCRIPT_DIR}/bin/${BINARY_NAME}"
if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: Binary not found: ${BINARY}"
    echo "Available binaries:"
    ls "${SCRIPT_DIR}/bin/"
    exit 1
fi

export TT_METAL_RUNTIME_ROOT="${SCRIPT_DIR}"
export LD_LIBRARY_PATH="${SCRIPT_DIR}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "${BINARY}" "${ARGS[@]}"
WRAPPER_EOF
chmod +x "${PKG_ROOT}/run_test.sh"

# --- Create README ---
cat > "${PKG_ROOT}/README" << 'README_EOF'
TT-Fabric Test Package
======================

Pre-built fabric test binaries for tt-metal.

Prerequisites on target machine:
  - SFPI toolchain at /opt/tenstorrent/sfpi (unless bundled with --bundle-sfpi)
  - Tenstorrent device drivers loaded
  - Compatible Linux (same distro/version as build machine recommended)

Single-host quick start:
  ./run_test.sh --test_config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_sanity_common.yaml

Run a different binary:
  ./run_test.sh --binary test_tt_fabric_mux_bandwidth

Multihost tests:
  Multihost tests use MPI to launch processes across multiple machines.

  Prerequisites (all hosts):
    - Package extracted to the same path (or on a shared NFS mount)
    - OpenMPI installed (mpirun / mpirun-ulfm)
    - Python 3.8+ with: pip install click pyyaml pydantic loguru
    - Passwordless SSH between hosts

  Using tt-run (included in package):
    export TT_METAL_HOME=$(pwd)
    export TT_METAL_RUNTIME_ROOT=$(pwd)
    export LD_LIBRARY_PATH=$(pwd)/lib
    python3 ttnn/ttnn/distributed/ttrun.py \
      --rank-binding tests/tt_metal/distributed/config/32x4_quad_bh_galaxy_rank_bindings.yaml \
      --mpi-args "--bind-to none --host host1,host2,host3,host4 --tag-output" \
      ./bin/test_tt_fabric \
      --test_config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_sanity_common.yaml

  Rank binding configs are in: tests/tt_metal/distributed/config/

Environment variables:
  TT_METAL_RUNTIME_ROOT  - Set automatically by run_test.sh to package root
  LD_LIBRARY_PATH        - Set automatically by run_test.sh to lib/
  TT_METAL_CACHE         - JIT cache directory (default: ~/.cache/tt-metal-cache/)
README_EOF

# --- Create tarball ---
echo "==> Creating tarball..."
tar -C "${STAGING_DIR}" -cf - tt-fabric-tests | zstd -T0 -3 -o "${OUTPUT}"

TARBALL_SIZE=$(du -sh "${OUTPUT}" | cut -f1)
echo ""
echo "=== Package created successfully ==="
echo "    Output:  ${OUTPUT}"
echo "    Size:    ${TARBALL_SIZE}"
echo ""
echo "To deploy:"
echo "    scp ${OUTPUT} user@target-machine:/path/"
echo "    ssh user@target-machine 'cd /path && zstd -dc tt-fabric-tests-*.tar.zst | tar xf -'"
echo "    ssh user@target-machine 'cd /path/tt-fabric-tests && ./run_test.sh --test_config tests/.../your_config.yaml'"
