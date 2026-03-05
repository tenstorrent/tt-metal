#!/usr/bin/env bash
# build-wrapper.sh - Thin wrapper around build-artifact.sh with common defaults.
# Provides a simplified interface for building tt-metal across configurations.
#
# Not itself an sbatch script — run directly from submit.sh or CI.
#
# Mirrors: .github/workflows/build-wrapper.yaml
#
# Usage:
#   ./slurm/workflows/build-wrapper.sh [options]
#
# Options:
#   --platform PLATFORM    "Ubuntu 22.04" or "Ubuntu 24.04"
#   --build-type TYPE      Release | Debug | RelWithDebInfo | ASan | TSan
#   --toolchain PATH       CMake toolchain file
#   --enable-lto           Enable link-time optimization
#   --image IMAGE          Pre-built Docker image tag
#   --timeout TIME         Build time limit (default: 02:00:00)
#   --matrix               Run the full multi-config matrix build
#
# In --matrix mode, submits 5 parallel builds matching the GHA matrix:
#   - Ubuntu 22.04 / clang-20 / Debug
#   - Ubuntu 22.04 / clang-20 libcpp / Release
#   - Ubuntu 22.04 / gcc-12 / Release
#   - Ubuntu 24.04 / clang-20 / Release + LTO
#   - Ubuntu 24.04 / gcc-14 / Release

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
source "${SCRIPT_DIR}/_helpers/submit_dependent.sh"
source_config env

require_env PIPELINE_ID

# ---------------------------------------------------------------------------
# Parse CLI arguments
# ---------------------------------------------------------------------------
PLATFORM=""
BUILD_TYPE=""
TOOLCHAIN=""
ENABLE_LTO="false"
DOCKER_IMAGE_OVERRIDE=""
BUILD_TIMEOUT="02:00:00"
MATRIX_MODE="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform)    PLATFORM="$2"; shift 2 ;;
        --build-type)  BUILD_TYPE="$2"; shift 2 ;;
        --toolchain)   TOOLCHAIN="$2"; shift 2 ;;
        --enable-lto)  ENABLE_LTO="true"; shift ;;
        --image)       DOCKER_IMAGE_OVERRIDE="$2"; shift 2 ;;
        --timeout)     BUILD_TIMEOUT="$2"; shift 2 ;;
        --matrix)      MATRIX_MODE="true"; shift ;;
        --arch)        export ARCH_NAME="$2"; shift 2 ;;
        --ref)         export GIT_REF="$2"; shift 2 ;;
        *)             log_warn "Unknown option: $1"; shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Matrix mode: submit all configurations in parallel
# ---------------------------------------------------------------------------
if [[ "${MATRIX_MODE}" == "true" ]]; then
    log_info "=== Build wrapper (matrix mode) ==="
    log_info "  Pipeline: ${PIPELINE_ID}"

    declare -a MATRIX_CONFIGS=(
        "Ubuntu 22.04|cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake|Debug|false"
        "Ubuntu 22.04|cmake/x86_64-linux-clang-20-libcpp-toolchain.cmake|Release|false"
        "Ubuntu 22.04|cmake/x86_64-linux-gcc-12-toolchain.cmake|Release|false"
        "Ubuntu 24.04|cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake|Release|true"
        "Ubuntu 24.04|cmake/x86_64-linux-gcc-14-toolchain.cmake|Release|false"
    )

    declare -a JOB_IDS=()
    for config in "${MATRIX_CONFIGS[@]}"; do
        IFS='|' read -r m_platform m_toolchain m_build_type m_lto <<< "${config}"

        EXPORT_VARS="ALL,PIPELINE_ID=${PIPELINE_ID}"
        EXPORT_VARS+=",PLATFORM=${m_platform}"
        EXPORT_VARS+=",TOOLCHAIN=${m_toolchain}"
        EXPORT_VARS+=",BUILD_TYPE=${m_build_type}"
        EXPORT_VARS+=",ENABLE_LTO=${m_lto}"
        EXPORT_VARS+=",PUBLISH_ARTIFACT=false"
        EXPORT_VARS+=",PUBLISH_PACKAGE=false"
        EXPORT_VARS+=",SKIP_TT_TRAIN=true"
        EXPORT_VARS+=",TRACY=false"
        [[ -n "${DOCKER_IMAGE_OVERRIDE}" ]] && EXPORT_VARS+=",DOCKER_IMAGE=${DOCKER_IMAGE_OVERRIDE}"

        # Derive a short name for the job
        tc_short="$(basename "${m_toolchain}" | sed -E 's/-toolchain\.cmake$//')"
        job_name="build-${m_build_type,,}-${tc_short}-${PIPELINE_ID}"

        JOB_ID="$(sbatch \
            --parsable \
            --job-name="${job_name}" \
            --partition=build \
            --time="${BUILD_TIMEOUT}" \
            --cpus-per-task=16 \
            --mem=64G \
            --output="logs/${job_name}-%j.out" \
            --export="${EXPORT_VARS}" \
            "${SCRIPT_DIR}/build-artifact.sh")"

        log_info "Submitted: ${m_platform} / ${m_build_type} / ${tc_short} -> JOBID=${JOB_ID}"
        JOB_IDS+=("${JOB_ID}")
    done

    log_info "=== Matrix build submitted: ${#JOB_IDS[@]} jobs ==="
    IFS=':' ; echo "${JOB_IDS[*]}"
    exit 0
fi

# ---------------------------------------------------------------------------
# Single build mode
# ---------------------------------------------------------------------------
PLATFORM="${PLATFORM:-Ubuntu 22.04}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

log_info "=== Build wrapper (single mode) ==="
log_info "  Pipeline:   ${PIPELINE_ID}"
log_info "  Platform:   ${PLATFORM}"
log_info "  Build type: ${BUILD_TYPE}"
log_info "  Toolchain:  ${TOOLCHAIN:-<default>}"
log_info "  LTO:        ${ENABLE_LTO}"
log_info "  Timeout:    ${BUILD_TIMEOUT}"

EXPORT_VARS="ALL,PIPELINE_ID=${PIPELINE_ID}"
EXPORT_VARS+=",PLATFORM=${PLATFORM}"
EXPORT_VARS+=",BUILD_TYPE=${BUILD_TYPE}"
EXPORT_VARS+=",ENABLE_LTO=${ENABLE_LTO}"
[[ -n "${TOOLCHAIN}" ]] && EXPORT_VARS+=",TOOLCHAIN=${TOOLCHAIN}"
[[ -n "${DOCKER_IMAGE_OVERRIDE}" ]] && EXPORT_VARS+=",DOCKER_IMAGE=${DOCKER_IMAGE_OVERRIDE}"

BUILD_JOB_ID="$(sbatch \
    --parsable \
    --job-name="build-${PIPELINE_ID}" \
    --partition=build \
    --time="${BUILD_TIMEOUT}" \
    --cpus-per-task=16 \
    --mem=64G \
    --output="logs/build-${PIPELINE_ID}-%j.out" \
    --export="${EXPORT_VARS}" \
    "${SCRIPT_DIR}/build-artifact.sh")"

log_info "Build submitted: JOBID=${BUILD_JOB_ID}"
echo "${BUILD_JOB_ID}"
