#!/bin/bash
# Local Docker Build Script for TT-Metalium
#
# Thin wrapper around `docker buildx bake` for local development.
# The build configuration lives in docker-bake.hcl (single source of truth).
#
# You can also call bake directly:
#   docker buildx bake -f dockerfile/docker-bake.hcl dev

show_help() {
    cat <<'EOF'
Local Docker Build Script for TT-Metalium

Usage:
  ./dockerfile/build-local.sh [OPTIONS] [TARGET]

Options:
  --ubuntu VERSION    Ubuntu version (default: 22.04)
  --tag TAG           Output image tag override
  --set KEY=VALUE     Additional bake --set override (repeatable)
  --no-cache          Build without Docker cache
  --print             Dry run: show what would be built
  --help              Show this help message

Targets:
  dev                 Development image (default)
  ci-build            CI build image
  ci-test             CI test image
  release             Release image
  release-models      Release models image
  basic-dev           Basic dev image
  basic-ttnn-runtime  Basic TTNN runtime image
  manylinux           ManyLinux image
  tools               All tool images only
  venvs               All Python venv images only
  all                 Everything

Examples:
  ./dockerfile/build-local.sh dev
  ./dockerfile/build-local.sh --ubuntu 24.04 ci-test
  ./dockerfile/build-local.sh --set ci-build.output=type=docker ci-build
  ./dockerfile/build-local.sh --no-cache dev
  ./dockerfile/build-local.sh --print dev

Or call bake directly:
  docker buildx bake -f dockerfile/docker-bake.hcl dev
  UBUNTU_VERSION=24.04 PYTHON_VERSION=3.12 docker buildx bake -f dockerfile/docker-bake.hcl dev
  docker buildx bake -f dockerfile/docker-bake.hcl --no-cache dev
  docker buildx bake -f dockerfile/docker-bake.hcl --print dev
EOF
    exit 0
}

set -euo pipefail

if ! docker buildx version &>/dev/null; then
    echo "ERROR: docker buildx is required but not available." >&2
    echo "See: https://docs.docker.com/build/install-buildx/" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BAKE_FILE="${SCRIPT_DIR}/docker-bake.hcl"

# Defaults
UBUNTU_VERSION="22.04"
PYTHON_VERSION="3.10"
TARGET="dev"
TAG_OVERRIDE=""
BAKE_FLAGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ubuntu)
            UBUNTU_VERSION="$2"
            if [ "$UBUNTU_VERSION" = "24.04" ]; then
                PYTHON_VERSION="3.12"
            fi
            shift 2
            ;;
        --tag)
            TAG_OVERRIDE="$2"
            shift 2
            ;;
        --set)
            BAKE_FLAGS+=("--set" "$2")
            shift 2
            ;;
        --no-cache)
            BAKE_FLAGS+=("--no-cache")
            shift
            ;;
        --print)
            BAKE_FLAGS+=("--print")
            shift
            ;;
        --help|-h)
            show_help
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            TARGET="$1"
            shift
            ;;
    esac
done

# Build the bake command
BAKE_ARGS=()
BAKE_ARGS+=(-f "$BAKE_FILE")
BAKE_ARGS+=("${BAKE_FLAGS[@]}")

if [ -n "$TAG_OVERRIDE" ]; then
    BAKE_ARGS+=(--set "${TARGET}.tags=${TAG_OVERRIDE}")
fi

BAKE_ARGS+=("$TARGET")

cd "$REPO_ROOT"
export UBUNTU_VERSION PYTHON_VERSION
exec docker buildx bake "${BAKE_ARGS[@]}"
