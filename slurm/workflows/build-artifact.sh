#!/usr/bin/env bash
#SBATCH --job-name=build-artifact
#SBATCH --partition=build
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#
# Central build job: cmake configure + compile inside the CI build Docker
# container, then package ttm_any.tar.zst and stage to Weka shared storage.
#
# Mirrors: .github/workflows/build-artifact.yaml
#
# Environment / flags:
#   PIPELINE_ID       (required) Pipeline identifier for artifact staging
#   PLATFORM          Platform string, default "Ubuntu 22.04"
#   BUILD_TYPE        Release | Debug | RelWithDebInfo | ASan | TSan
#   TOOLCHAIN         CMake toolchain file path
#   TRACY             true/false — build with profiler support
#   DISTRIBUTED       true/false — build with OpenMPI (distributed)
#   SKIP_TT_TRAIN     true/false — skip tt-train targets
#   ENABLE_LTO        true/false — link-time optimization
#   BUILD_UMD_TESTS   true/false — build UMD test suite
#   PROFILE           true/false — ClangBuildAnalyzer profiling
#   PUBLISH_ARTIFACT  true/false — stage tarball to shared storage (default true)
#   PUBLISH_PACKAGE   true/false — stage .deb packages (default true)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib artifacts
source_lib docker
source_config env

source "${SCRIPT_DIR}/_helpers/resolve_docker_image.sh"

require_env PIPELINE_ID

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
PLATFORM="${PLATFORM:-Ubuntu 22.04}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
TRACY="${TRACY:-false}"
DISTRIBUTED="${DISTRIBUTED:-true}"
SKIP_TT_TRAIN="${SKIP_TT_TRAIN:-true}"
ENABLE_LTO="${ENABLE_LTO:-false}"
BUILD_UMD_TESTS="${BUILD_UMD_TESTS:-false}"
PROFILE="${PROFILE:-false}"
PUBLISH_ARTIFACT="${PUBLISH_ARTIFACT:-true}"
PUBLISH_PACKAGE="${PUBLISH_PACKAGE:-true}"

# ---------------------------------------------------------------------------
# Parse platform -> distro, version, toolchain, python version
# ---------------------------------------------------------------------------
case "${PLATFORM}" in
    "Ubuntu 24.04")
        DISTRO="ubuntu"; VERSION="24.04"; PYTHON_VERSION="3.12"
        TOOLCHAIN="${TOOLCHAIN:-cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake}"
        ;;
    "Ubuntu 22.04"|*)
        DISTRO="ubuntu"; VERSION="22.04"; PYTHON_VERSION="3.10"
        TOOLCHAIN="${TOOLCHAIN:-cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake}"
        ;;
esac

ARCH="${DOCKER_IMAGE_ARCH:-amd64}"

log_info "=== Build artifact starting ==="
log_info "  Pipeline:    ${PIPELINE_ID}"
log_info "  Platform:    ${PLATFORM} (${DISTRO} ${VERSION})"
log_info "  Build type:  ${BUILD_TYPE}"
log_info "  Toolchain:   ${TOOLCHAIN}"
log_info "  Tracy:       ${TRACY}"
log_info "  Distributed: ${DISTRIBUTED}"
log_info "  LTO:         ${ENABLE_LTO}"

# ---------------------------------------------------------------------------
# Resolve Docker image
# ---------------------------------------------------------------------------
resolve_workflow_docker_image "ci-build"
log_info "  Docker image: ${DOCKER_IMAGE}"

docker_login
docker_pull_with_retry "${DOCKER_IMAGE}"

# ---------------------------------------------------------------------------
# Build inside container
# ---------------------------------------------------------------------------
NPROC="${SLURM_CPUS_PER_TASK:-16}"

BUILD_COMMANDS="
cd \${TT_METAL_HOME}
git config --global --add safe.directory \${TT_METAL_HOME}

# --- ccache setup ---
mkdir -p /tmp/ccache
export CCACHE_REMOTE_ONLY=true
export CCACHE_TEMPDIR=/tmp/ccache
export TRACY_NO_INVARIANT_CHECK=1
export TRACY_NO_ISA_EXTENSIONS=1

# Load Redis ccache config if available
if [[ -n \"\${CCACHE_REMOTE_STORAGE:-}\" ]]; then
    echo \"ccache remote storage configured\"
fi
ccache -z

# --- Configure ---
BUILD_ARGS=\"\"
if [[ \"${SKIP_TT_TRAIN}\" == \"true\" ]]; then
    BUILD_ARGS=\"--build-metal-tests --build-ttnn-tests --build-programming-examples\"
else
    BUILD_ARGS=\"--build-all\"
fi

build_cmd=(
    ./build_metal.sh
    --build-dir build
    --build-type ${BUILD_TYPE}
    --toolchain-path ${TOOLCHAIN}
    \${BUILD_ARGS}
    --enable-ccache
    --configure-only
)

if [[ \"${TRACY}\" == \"false\" ]]; then
    build_cmd+=(--disable-profiler)
fi
if [[ \"${DISTRIBUTED}\" == \"false\" ]]; then
    build_cmd+=(--without-distributed)
fi
if [[ \"${PROFILE}\" == \"true\" ]]; then
    build_cmd+=(--enable-time-trace --disable-unity-builds)
fi
if [[ \"${BUILD_UMD_TESTS}\" == \"true\" ]]; then
    build_cmd+=(--build-umd-tests)
fi
if [[ \"${ENABLE_LTO}\" == \"true\" ]]; then
    build_cmd+=(--enable-lto)
fi

echo \"Running: \${build_cmd[*]}\"
\"\${build_cmd[@]}\"

# --- Compile ---
cmake --build build --target install -- -j${NPROC}

# --- Package (.deb) ---
cmake --build build --target package || true
ls -1sh build/*.deb build/*.ddeb 2>/dev/null || true

# --- Build profile (optional) ---
if [[ \"${PROFILE}\" == \"true\" ]]; then
    echo 'maxNameLength = 300' > ClangBuildAnalyzer.ini
    ClangBuildAnalyzer --all build capture.bin
    ClangBuildAnalyzer --analyze capture.bin
fi

# --- ccache summary ---
echo '=== CCache Summary ==='
ccache -s

# --- Create tarball ---
ARTIFACT_PATHS=\"ttnn/ttnn/*.so build/lib build/programming_examples build/test build/tools runtime\"
if [[ \"${SKIP_TT_TRAIN}\" != \"true\" ]]; then
    ARTIFACT_PATHS=\"\${ARTIFACT_PATHS} build/tt-train data\"
fi
tar -I 'zstd --adapt=min=9 -T0' -cvhf \${TT_METAL_HOME}/build/ttm_any.tar.zst \${ARTIFACT_PATHS}
"

DOCKER_EXTRA_ENV="PIPELINE_ID=${PIPELINE_ID}
CMAKE_BUILD_PARALLEL_LEVEL=${NPROC}
ARCH_NAME=${ARCH_NAME}
CCACHE_REMOTE_STORAGE=${CCACHE_REMOTE_STORAGE:-}"
export DOCKER_EXTRA_ENV

DOCKER_EXTRA_VOLUMES="${HOME}/.ccache-ci:/github/home/.ccache"
export DOCKER_EXTRA_VOLUMES

DOCKER_EXTRA_OPTS="--group-add 1457 --tmpfs /tmp"
export DOCKER_EXTRA_OPTS

docker_run "${DOCKER_IMAGE}" "${BUILD_COMMANDS}"

# ---------------------------------------------------------------------------
# Stage artifacts to shared storage
# ---------------------------------------------------------------------------
if [[ "${PUBLISH_ARTIFACT}" == "true" ]]; then
    log_info "Staging build tarball to shared storage"
    stage_build_artifact "${PIPELINE_ID}" "${REPO_ROOT}/build"
fi

if [[ "${PUBLISH_PACKAGE}" == "true" ]]; then
    local_pkg_dir="${REPO_ROOT}/build"
    artifact_dir="$(get_artifact_dir "${PIPELINE_ID}")/packages"
    mkdir -p "${artifact_dir}"

    for pkg in "${local_pkg_dir}"/*.deb "${local_pkg_dir}"/*.ddeb; do
        [[ -f "${pkg}" ]] || continue
        log_info "Staging package: $(basename "${pkg}")"
        cp "${pkg}" "${artifact_dir}/"
    done
fi

log_info "=== Build artifact complete ==="
