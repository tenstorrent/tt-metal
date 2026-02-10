#!/bin/bash

show_help() {
    cat <<'EOF'
Local Docker Build Script for TT-Metalium

This script automates building the TT-Metalium Docker images locally by:
1. Building any missing tool images (from Dockerfile.tools)
2. Building any missing Python venv images (from Dockerfile.python)
3. Building the main image with all dependencies

Usage:
  ./dockerfile/build-local.sh [OPTIONS] [TARGET]

Options:
  --ubuntu VERSION    Ubuntu version (default: 22.04)
  --tag TAG           Output image tag (default: tt-metalium-<target>:local)
  --rebuild-tools     Force rebuild of tool images even if they exist
  --rebuild-venvs     Force rebuild of venv images even if they exist
  --rebuild-all       Force rebuild of everything
  --no-cache          Build without Docker cache
  --help              Show this help message

Targets:
  ci-build            CI build image
  ci-test             CI test image
  dev                 Development image (default)
  release             Release image
  release-models      Release models image

Examples:
  ./dockerfile/build-local.sh dev
  ./dockerfile/build-local.sh --ubuntu 24.04 ci-test
  ./dockerfile/build-local.sh --rebuild-tools dev
EOF
    exit 0
}

set -euo pipefail

# Determine script and repo root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default configuration
UBUNTU_VERSION="22.04"
TARGET="dev"
OUTPUT_TAG=""
REBUILD_TOOLS=false
REBUILD_VENVS=false
NO_CACHE=""

# Tool images configuration
# Format: "target:local-tag"
TOOL_IMAGES=(
    "ccache:tool-ccache:local"
    "mold:tool-mold:local"
    "doxygen:tool-doxygen:local"
    "cba:tool-cba:local"
    "gdb:tool-gdb:local"
    "cmake:tool-cmake:local"
    "yq:tool-yq:local"
    "sfpi:tool-sfpi:local"
    "openmpi:tool-openmpi:local"
)

# Python venv images configuration
# Format: "target:local-tag"
VENV_IMAGES=(
    "ci-build-venv:python-ci-build-venv:local"
    "ci-test-venv:python-ci-test-venv:local"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Check if a Docker image exists locally
image_exists() {
    local image="$1"
    docker image inspect "$image" &>/dev/null
}

# Build a tool image if it doesn't exist (or if rebuild is forced)
build_tool_image() {
    local target="$1"
    local tag="$2"

    if [ "$REBUILD_TOOLS" = true ] || ! image_exists "$tag"; then
        log_info "Building tool image: $tag (target: $target)"
        docker build \
            -f "${SCRIPT_DIR}/Dockerfile.tools" \
            --target "$target" \
            -t "$tag" \
            $NO_CACHE \
            "${REPO_ROOT}"
        log_success "Built $tag"
    else
        log_success "Tool image exists: $tag"
    fi
}

# Build a Python venv image if it doesn't exist (or if rebuild is forced)
build_venv_image() {
    local target="$1"
    local tag="$2"

    if [ "$REBUILD_VENVS" = true ] || ! image_exists "$tag"; then
        log_info "Building venv image: $tag (target: $target, Ubuntu: $UBUNTU_VERSION)"
        docker build \
            -f "${SCRIPT_DIR}/Dockerfile.python" \
            --target "$target" \
            --build-arg "UBUNTU_VERSION=${UBUNTU_VERSION}" \
            -t "$tag" \
            $NO_CACHE \
            "${REPO_ROOT}"
        log_success "Built $tag"
    else
        log_success "Venv image exists: $tag"
    fi
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --ubuntu)
                UBUNTU_VERSION="$2"
                shift 2
                ;;
            --tag)
                OUTPUT_TAG="$2"
                shift 2
                ;;
            --rebuild-tools)
                REBUILD_TOOLS=true
                shift
                ;;
            --rebuild-venvs)
                REBUILD_VENVS=true
                shift
                ;;
            --rebuild-all)
                REBUILD_TOOLS=true
                REBUILD_VENVS=true
                shift
                ;;
            --no-cache)
                NO_CACHE="--no-cache"
                shift
                ;;
            --help|-h)
                show_help
                ;;
            -*)
                log_error "Unknown option: $1"
                exit 1
                ;;
            *)
                TARGET="$1"
                shift
                ;;
        esac
    done

    # Set default output tag if not specified
    if [ -z "$OUTPUT_TAG" ]; then
        OUTPUT_TAG="tt-metalium-${TARGET}:local"
    fi
}

# Main build process
main() {
    parse_args "$@"

    log_info "TT-Metalium Local Build"
    log_info "  Target: $TARGET"
    log_info "  Ubuntu: $UBUNTU_VERSION"
    log_info "  Output: $OUTPUT_TAG"
    echo ""

    # Step 1: Build tool images
    log_info "=== Step 1/3: Checking tool images ==="
    for spec in "${TOOL_IMAGES[@]}"; do
        IFS=':' read -r target tag <<< "$spec"
        build_tool_image "$target" "$tag"
    done
    echo ""

    # Step 2: Build Python venv images
    log_info "=== Step 2/3: Checking Python venv images ==="
    for spec in "${VENV_IMAGES[@]}"; do
        IFS=':' read -r target tag <<< "$spec"
        build_venv_image "$target" "$tag"
    done
    echo ""

    # Step 3: Build the main image
    log_info "=== Step 3/3: Building main image ==="
    log_info "Building: $OUTPUT_TAG (target: $TARGET)"

    docker build \
        -f "${SCRIPT_DIR}/Dockerfile" \
        --target "$TARGET" \
        --build-arg "UBUNTU_VERSION=${UBUNTU_VERSION}" \
        --build-arg "TOOL_CCACHE_IMAGE=tool-ccache:local" \
        --build-arg "TOOL_MOLD_IMAGE=tool-mold:local" \
        --build-arg "TOOL_DOXYGEN_IMAGE=tool-doxygen:local" \
        --build-arg "TOOL_CBA_IMAGE=tool-cba:local" \
        --build-arg "TOOL_GDB_IMAGE=tool-gdb:local" \
        --build-arg "TOOL_CMAKE_IMAGE=tool-cmake:local" \
        --build-arg "TOOL_YQ_IMAGE=tool-yq:local" \
        --build-arg "TOOL_SFPI_IMAGE=tool-sfpi:local" \
        --build-arg "TOOL_OPENMPI_IMAGE=tool-openmpi:local" \
        --build-arg "PYTHON_CI_BUILD_VENV_IMAGE=python-ci-build-venv:local" \
        --build-arg "PYTHON_CI_TEST_VENV_IMAGE=python-ci-test-venv:local" \
        -t "$OUTPUT_TAG" \
        $NO_CACHE \
        "${REPO_ROOT}"

    echo ""
    log_success "Build complete: $OUTPUT_TAG"
    echo ""
    log_info "To run the image:"
    echo "  docker run -it --rm $OUTPUT_TAG"
}

main "$@"
