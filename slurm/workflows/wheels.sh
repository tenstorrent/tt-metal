#!/usr/bin/env bash
#SBATCH --job-name=build-wheels
#SBATCH --partition=build
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#
# Build Python wheels using cibuildwheel inside a manylinux Docker container.
#
# Mirrors: .github/workflows/wheels.yaml
#
# Environment / flags:
#   PIPELINE_ID       (required) Pipeline identifier for artifact staging
#   PYTHON_VERSION    Python version to target (default: 3.10)
#   BUILD_TYPE        Release | Debug (default: Release)
#   TRACY             true/false — profiler support in wheel
#   ENABLE_LTO        true/false — link-time optimization
#   REF               Git ref to checkout (default: HEAD)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source_lib artifacts
source_lib docker
source_config env

source "${SCRIPT_DIR}/_helpers/resolve_docker_image.sh"

require_env PIPELINE_ID

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PYTHON_VERSION_NODOT="${PYTHON_VERSION//./}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
TRACY="${TRACY:-true}"
ENABLE_LTO="${ENABLE_LTO:-false}"
REF="${REF:-${GIT_SHA}}"

log_info "=== Wheel build starting ==="
log_info "  Pipeline:       ${PIPELINE_ID}"
log_info "  Python version: ${PYTHON_VERSION} (cp${PYTHON_VERSION_NODOT})"
log_info "  Build type:     ${BUILD_TYPE}"
log_info "  Tracy:          ${TRACY}"
log_info "  LTO:            ${ENABLE_LTO}"

# ---------------------------------------------------------------------------
# Resolve manylinux Docker image
# ---------------------------------------------------------------------------
resolve_workflow_docker_image "manylinux"
MANYLINUX_IMAGE="${DOCKER_IMAGE}"
log_info "  Image: ${MANYLINUX_IMAGE}"

docker_login
docker_pull_with_retry "${MANYLINUX_IMAGE}"

# ---------------------------------------------------------------------------
# Build wheel via cibuildwheel inside container
# ---------------------------------------------------------------------------
NPROC="${SLURM_CPUS_PER_TASK:-16}"
WHEEL_DIR="${REPO_ROOT}/wheelhouse"
mkdir -p "${WHEEL_DIR}"

TRACY_FLAG="OFF"
[[ "${TRACY}" == "true" ]] && TRACY_FLAG="ON"
LTO_FLAG="OFF"
[[ "${ENABLE_LTO}" == "true" ]] && LTO_FLAG="ON"

WHEEL_COMMANDS="
cd \${TT_METAL_HOME}
git config --global --add safe.directory \${TT_METAL_HOME}

# ccache setup
mkdir -p /tmp/ccache
export CCACHE_REMOTE_ONLY=true
export CCACHE_TEMPDIR=/tmp/ccache
if [[ -n \"\${CCACHE_REMOTE_STORAGE:-}\" ]]; then
    echo \"ccache remote storage configured\"
fi
ccache -z

# Use cibuildwheel if available, otherwise fall back to pip wheel
if command -v cibuildwheel &>/dev/null; then
    export CIBW_BUILD=\"cp${PYTHON_VERSION_NODOT}-manylinux_x86_64*\"
    export CIBW_SKIP=\"*-musllinux_*\"
    export CIBW_BUILD_FRONTEND=build
    export CIBW_ENVIRONMENT=\"CCACHE_REMOTE_ONLY=true CCACHE_TEMPDIR=/tmp/ccache CIBW_ENABLE_TRACY=${TRACY_FLAG} CIBW_BUILD_TYPE=${BUILD_TYPE} CIBW_ENABLE_LTO=${LTO_FLAG}\"
    export CIBW_BEFORE_BUILD=\"mkdir -p /tmp/ccache && ccache -z && ccache -p\"
    export CIBW_BEFORE_TEST=\"ccache -s\"
    export CIBW_TEST_COMMAND='python -c \"import ttnn\"'
    export CIBW_MANYLINUX_X86_64_IMAGE=\"${MANYLINUX_IMAGE}\"
    cibuildwheel --output-dir \${TT_METAL_HOME}/wheelhouse
else
    pip wheel . --no-deps --wheel-dir \${TT_METAL_HOME}/wheelhouse/
fi

ccache -s
ls -lh \${TT_METAL_HOME}/wheelhouse/
"

DOCKER_EXTRA_ENV="PIPELINE_ID=${PIPELINE_ID}
ARCH_NAME=${ARCH_NAME}
CCACHE_REMOTE_STORAGE=${CCACHE_REMOTE_STORAGE:-}"
export DOCKER_EXTRA_ENV

docker_run "${MANYLINUX_IMAGE}" "${WHEEL_COMMANDS}"

# ---------------------------------------------------------------------------
# Stage wheel to shared storage
# ---------------------------------------------------------------------------
for whl in "${WHEEL_DIR}"/*.whl; do
    [[ -f "${whl}" ]] || continue
    log_info "Staging wheel: $(basename "${whl}")"
    stage_wheel "${PIPELINE_ID}" "${whl}"
done

# ---------------------------------------------------------------------------
# Compute artifact name for downstream reference
# ---------------------------------------------------------------------------
SANITIZED_REF="$(echo "${REF}" | sed 's/[\/:"<>|*?\r\n\\]/-/g')"
TRACY_SUFFIX=""
[[ "${TRACY}" == "true" ]] && TRACY_SUFFIX="-profiler"
ARTIFACT_NAME="ttnn-dist-cp${PYTHON_VERSION_NODOT}-${BUILD_TYPE}${TRACY_SUFFIX}-${SANITIZED_REF}-${PIPELINE_ID}"
log_info "Wheel artifact name: ${ARTIFACT_NAME}"

log_info "=== Wheel build complete ==="
