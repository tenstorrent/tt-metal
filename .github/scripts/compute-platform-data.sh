#!/bin/bash
# Compute platform-specific Docker image tags and check existence
#
# Usage: compute-platform-data.sh <version> <repo> [--force-rebuild] [--check-exists]
#
# Arguments:
#   version       Ubuntu version (e.g., "22.04" or "24.04")
#   repo          GitHub repository (e.g., "owner/repo")
#   --force-rebuild  Treat all images as missing (optional)
#   --check-exists   Check if images exist in registry (optional, default: true)
#
# Output: JSON object with tags, existence flags, and metadata
#
# Example:
#   compute-platform-data.sh "22.04" "tenstorrent/tt-metal" --check-exists

set -euo pipefail

usage() {
    echo "Usage: $0 <version> <repo> [--force-rebuild] [--check-exists]" >&2
    exit 1
}

if [[ $# -lt 2 ]]; then
    usage
fi

VERSION="$1"
REPO="$2"
shift 2

FORCE_REBUILD=false
CHECK_EXISTS=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force-rebuild)
            FORCE_REBUILD=true
            shift
            ;;
        --check-exists)
            CHECK_EXISTS=true
            shift
            ;;
        --no-check-exists)
            CHECK_EXISTS=false
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
done

DISTRO="ubuntu"
ARCH="amd64"

# Determine Python version based on Ubuntu version
if [ "$VERSION" = "24.04" ]; then
    PYTHON_VERSION="3.12"
else
    PYTHON_VERSION="3.10"
fi

VERSION_NODOT="${VERSION//.}"

# Image names
CI_BUILD_NAME="${DISTRO}-${VERSION}-ci-build-${ARCH}"
CI_TEST_NAME="${DISTRO}-${VERSION}-ci-test-${ARCH}"
DEV_NAME="${DISTRO}-${VERSION}-dev-${ARCH}"
BASIC_DEV_NAME="${DISTRO}-${VERSION}-basic-dev-${ARCH}"
BASIC_TTNN_NAME="${DISTRO}-${VERSION}-basic-ttnn-runtime-${ARCH}"

# Extra files for hash computation
EXTRA_FILES=".github/workflows/build-docker-artifact.yaml dockerfile/Dockerfile.tools"
BASIC_DEV_EXTRA_FILES="$EXTRA_FILES"
MANYLINUX_EXTRA_FILES="$EXTRA_FILES"

# Compute hashes
HASH=$(.github/scripts/dockerfile-hash.sh dockerfile/Dockerfile $EXTRA_FILES)
BASIC_DEV_HASH=$(.github/scripts/dockerfile-hash.sh dockerfile/Dockerfile.basic-dev $BASIC_DEV_EXTRA_FILES)
BASIC_TTNN_HASH="$BASIC_DEV_HASH"
MANYLINUX_HASH=$(.github/scripts/dockerfile-hash.sh dockerfile/Dockerfile.manylinux $MANYLINUX_EXTRA_FILES)
VENV_HASH=$(.github/scripts/dockerfile-hash.sh dockerfile/Dockerfile.python)

# Build tags
CI_BUILD_TAG="ghcr.io/${REPO}/tt-metalium/${CI_BUILD_NAME}:${HASH}"
CI_TEST_TAG="ghcr.io/${REPO}/tt-metalium/${CI_TEST_NAME}:${HASH}"
DEV_TAG="ghcr.io/${REPO}/tt-metalium/${DEV_NAME}:${HASH}"
BASIC_DEV_TAG="ghcr.io/${REPO}/tt-metalium/${BASIC_DEV_NAME}:${BASIC_DEV_HASH}"
BASIC_TTNN_TAG="ghcr.io/${REPO}/tt-metalium/${BASIC_TTNN_NAME}:${BASIC_TTNN_HASH}"
MANYLINUX_TAG="ghcr.io/${REPO}/tt-metalium/manylinux-${ARCH}:${MANYLINUX_HASH}"

BASE_VENV="ghcr.io/${REPO}/tt-metalium/python-venv"
CI_BUILD_VENV_TAG="${BASE_VENV}/ci-build:${VERSION_NODOT}-${VENV_HASH}"
CI_TEST_VENV_TAG="${BASE_VENV}/ci-test:${VERSION_NODOT}-${VENV_HASH}"

# Check existence (or set all to false if force-rebuild)
if [ "$FORCE_REBUILD" = "true" ]; then
    DEV_EXISTS=false
    BASIC_DEV_EXISTS=false
    BASIC_TTNN_EXISTS=false
    MANYLINUX_EXISTS=false
    CI_BUILD_VENV_EXISTS=false
    CI_TEST_VENV_EXISTS=false
elif [ "$CHECK_EXISTS" = "true" ]; then
    docker manifest inspect "$DEV_TAG" > /dev/null 2>&1 && DEV_EXISTS=true || DEV_EXISTS=false
    docker manifest inspect "$BASIC_DEV_TAG" > /dev/null 2>&1 && BASIC_DEV_EXISTS=true || BASIC_DEV_EXISTS=false
    docker manifest inspect "$BASIC_TTNN_TAG" > /dev/null 2>&1 && BASIC_TTNN_EXISTS=true || BASIC_TTNN_EXISTS=false
    docker manifest inspect "$MANYLINUX_TAG" > /dev/null 2>&1 && MANYLINUX_EXISTS=true || MANYLINUX_EXISTS=false
    docker manifest inspect "$CI_BUILD_VENV_TAG" > /dev/null 2>&1 && CI_BUILD_VENV_EXISTS=true || CI_BUILD_VENV_EXISTS=false
    docker manifest inspect "$CI_TEST_VENV_TAG" > /dev/null 2>&1 && CI_TEST_VENV_EXISTS=true || CI_TEST_VENV_EXISTS=false
else
    DEV_EXISTS=unknown
    BASIC_DEV_EXISTS=unknown
    BASIC_TTNN_EXISTS=unknown
    MANYLINUX_EXISTS=unknown
    CI_BUILD_VENV_EXISTS=unknown
    CI_TEST_VENV_EXISTS=unknown
fi

# ci-build and ci-test existence mirrors dev existence
CI_BUILD_EXISTS="$DEV_EXISTS"
CI_TEST_EXISTS="$DEV_EXISTS"

# Output JSON
# Use --argjson for boolean fields to output proper JSON booleans (true/false) not strings
jq -cn \
    --arg distro "$DISTRO" \
    --arg version "$VERSION" \
    --arg python_version "$PYTHON_VERSION" \
    --arg ci_build_tag "$CI_BUILD_TAG" \
    --arg ci_test_tag "$CI_TEST_TAG" \
    --arg dev_tag "$DEV_TAG" \
    --arg basic_dev_tag "$BASIC_DEV_TAG" \
    --arg basic_ttnn_tag "$BASIC_TTNN_TAG" \
    --arg manylinux_tag "$MANYLINUX_TAG" \
    --arg ci_build_venv_tag "$CI_BUILD_VENV_TAG" \
    --arg ci_test_venv_tag "$CI_TEST_VENV_TAG" \
    --argjson ci_build_exists "$CI_BUILD_EXISTS" \
    --argjson ci_test_exists "$CI_TEST_EXISTS" \
    --argjson dev_exists "$DEV_EXISTS" \
    --argjson basic_dev_exists "$BASIC_DEV_EXISTS" \
    --argjson basic_ttnn_exists "$BASIC_TTNN_EXISTS" \
    --argjson manylinux_exists "$MANYLINUX_EXISTS" \
    --argjson ci_build_venv_exists "$CI_BUILD_VENV_EXISTS" \
    --argjson ci_test_venv_exists "$CI_TEST_VENV_EXISTS" \
    '{
        distro: $distro,
        version: $version,
        python_version: $python_version,
        ci_build_tag: $ci_build_tag,
        ci_test_tag: $ci_test_tag,
        dev_tag: $dev_tag,
        basic_dev_tag: $basic_dev_tag,
        basic_ttnn_tag: $basic_ttnn_tag,
        manylinux_tag: $manylinux_tag,
        ci_build_venv_tag: $ci_build_venv_tag,
        ci_test_venv_tag: $ci_test_venv_tag,
        ci_build_exists: $ci_build_exists,
        ci_test_exists: $ci_test_exists,
        dev_exists: $dev_exists,
        basic_dev_exists: $basic_dev_exists,
        basic_ttnn_exists: $basic_ttnn_exists,
        manylinux_exists: $manylinux_exists,
        ci_build_venv_exists: $ci_build_venv_exists,
        ci_test_venv_exists: $ci_test_venv_exists
    }'
