# Docker Bake file for TT-Metalium
# =============================================================================
# Single source of truth for all Docker image build targets, replacing the
# build-local.sh script and centralizing build configuration that was
# previously duplicated across shell scripts, GitHub workflows, and
# composite actions.
#
# References:
# https://docs.docker.com/build/bake/reference/
# https://docs.docker.com/guides/bake/
# https://docs.docker.com/reference/cli/docker/buildx/bake/
#
# USAGE (local development):
#   docker buildx bake dev              # Build dev image (tools+venvs built automatically)
#   docker buildx bake ci-test          # Build CI test image
#   docker buildx bake ci-build         # Build CI build image
#   docker buildx bake basic-dev        # Build basic dev image
#   docker buildx bake manylinux        # Build manylinux image
#   docker buildx bake tools            # Build just the tool images
#   docker buildx bake venvs            # Build just the venv images
#   docker buildx bake all              # Build everything
#   docker buildx bake --print dev      # Dry run: show what would be built
#
#   # Ubuntu 24.04 variant:
#   UBUNTU_VERSION=24.04 PYTHON_VERSION=3.12 docker buildx bake dev
#
#   # Force rebuild (no cache):
#   docker buildx bake --no-cache dev
#
# USAGE (CI / GHCR overrides):
#   Workflows override tool/venv contexts to point to pre-built GHCR images and
#   invoke bake via .github/actions/manual-docker-bake (manual CLI instead of
#   docker/bake-action, which had registry metadata timeout issues).
#     docker buildx bake ci-build ci-test dev \
#       --set 'ci-build.contexts.ccache-layer=docker-image://ghcr.io/.../ccache:tag' \
#       --set 'ci-build.contexts.mold-layer=docker-image://ghcr.io/.../mold:tag' \
#       ...
#
# ARCHITECTURE:
#   Tool targets (Dockerfile.tools)  ──┐
#   Venv targets (Dockerfile.python) ──┼──> Main targets (Dockerfile)
#                                      ├──> Basic targets (Dockerfile.basic-dev)
#                                      └──> Manylinux target (Dockerfile.manylinux)
#
# The `contexts` feature wires tool/venv outputs into downstream builds:
#   - Locally: "target:ccache" means bake builds the ccache target first
#   - In CI:   override to "docker-image://ghcr.io/.../ccache:tag"
# =============================================================================

# =============================================================================
# Variables
# =============================================================================

variable "UBUNTU_VERSION" {
  default = "22.04"
}

variable "PYTHON_VERSION" {
  # Must match UBUNTU_VERSION: 22.04 -> 3.10, 24.04 -> 3.12
  default = "3.10"
}

variable "UV_IMAGE" {
  # SINGLE SOURCE OF TRUTH for the uv image SHA256 digest.
  # When upgrading uv, update this value and the fallback defaults in:
  #   - dockerfile/Dockerfile        (ARG UV_IMAGE)
  #   - dockerfile/Dockerfile.python  (ARG UV_IMAGE)
  # The Dockerfile defaults are only used for standalone `docker build` without
  # Bake; Bake always passes this variable, so it takes precedence.
  # See: https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
  default = "ghcr.io/astral-sh/uv@sha256:9a23023be68b2ed09750ae636228e903a54a05ea56ed03a934d00fe9fbeded4b"
}

# =============================================================================
# Tool targets (from Dockerfile.tools)
#
# Each tool is built as a separate multi-stage target that outputs pre-built
# binaries to /install/. These are consumed by downstream Dockerfiles via
# COPY --from or RUN --mount=from.
#
# Tool versions and SHA256 hashes are defined in Dockerfile.tools (the single
# source of truth for tool configuration).
# =============================================================================

target "ccache" {
  context    = "."
  dockerfile = "dockerfile/Dockerfile.tools"
  target     = "ccache"
  tags       = ["tool-ccache:local"]
}

target "mold" {
  context    = "."
  dockerfile = "dockerfile/Dockerfile.tools"
  target     = "mold"
  tags       = ["tool-mold:local"]
}

target "doxygen" {
  context    = "."
  dockerfile = "dockerfile/Dockerfile.tools"
  target     = "doxygen"
  tags       = ["tool-doxygen:local"]
}

target "cba" {
  context    = "."
  dockerfile = "dockerfile/Dockerfile.tools"
  target     = "cba"
  tags       = ["tool-cba:local"]
}

target "gdb" {
  context    = "."
  dockerfile = "dockerfile/Dockerfile.tools"
  target     = "gdb"
  tags       = ["tool-gdb:local"]
}

target "cmake" {
  context    = "."
  dockerfile = "dockerfile/Dockerfile.tools"
  target     = "cmake"
  tags       = ["tool-cmake:local"]
}

target "yq" {
  context    = "."
  dockerfile = "dockerfile/Dockerfile.tools"
  target     = "yq"
  tags       = ["tool-yq:local"]
}

target "sfpi" {
  context    = "."
  dockerfile = "dockerfile/Dockerfile.tools"
  target     = "sfpi"
  tags       = ["tool-sfpi:local"]
}

target "openmpi" {
  context    = "."
  dockerfile = "dockerfile/Dockerfile.tools"
  target     = "openmpi"
  tags       = ["tool-openmpi:local"]
}

group "tools" {
  targets = ["ccache", "mold", "doxygen", "cba", "gdb", "cmake", "yq", "sfpi", "openmpi"]
}

# =============================================================================
# Python venv targets (from Dockerfile.python)
#
# Pre-built virtual environments with all Python dependencies installed.
# Using pre-built venvs saves 5-10 minutes per build by avoiding pip installs.
# =============================================================================

target "ci-build-venv" {
  context    = "."
  dockerfile = "dockerfile/Dockerfile.python"
  target     = "ci-build-venv"
  args = {
    UBUNTU_VERSION = UBUNTU_VERSION
    PYTHON_VERSION = PYTHON_VERSION
    UV_IMAGE       = UV_IMAGE
  }
  tags = ["python-ci-build-venv:local"]
}

target "ci-test-venv" {
  context    = "."
  dockerfile = "dockerfile/Dockerfile.python"
  target     = "ci-test-venv"
  args = {
    UBUNTU_VERSION = UBUNTU_VERSION
    PYTHON_VERSION = PYTHON_VERSION
    UV_IMAGE       = UV_IMAGE
  }
  tags = ["python-ci-test-venv:local"]
}

group "venvs" {
  targets = ["ci-build-venv", "ci-test-venv"]
}

# =============================================================================
# Main image targets (from Dockerfile)
#
# Use Bake's `contexts` to wire in tool and venv layers. Bake builds the
# referenced targets automatically before the main image.
#
# The context names (ccache-layer, mold-layer, etc.) match the stage names
# in the Dockerfile. Bake overrides those stages with the tool target outputs.
# =============================================================================

target "_main-common" {
  context    = "."
  dockerfile = "dockerfile/Dockerfile"
  args = {
    UBUNTU_VERSION = UBUNTU_VERSION
    PYTHON_VERSION = PYTHON_VERSION
    UV_IMAGE       = UV_IMAGE
  }
  contexts = {
    # Tool layers (resolved from Dockerfile.tools targets locally)
    ccache-layer         = "target:ccache"
    mold-layer           = "target:mold"
    doxygen-layer        = "target:doxygen"
    cba-layer            = "target:cba"
    gdb-layer            = "target:gdb"
    cmake-layer          = "target:cmake"
    yq-layer             = "target:yq"
    sfpi-layer           = "target:sfpi"
    openmpi-layer        = "target:openmpi"
    # Python venv layers (resolved from Dockerfile.python targets locally)
    ci-build-venv-layer  = "target:ci-build-venv"
    ci-test-venv-layer   = "target:ci-test-venv"
  }
}

target "ci-build" {
  inherits = ["_main-common"]
  target   = "ci-build"
  tags     = ["tt-metalium-ci-build:local"]
}

target "ci-test" {
  inherits = ["_main-common"]
  target   = "ci-test"
  tags     = ["tt-metalium-ci-test:local"]
}

target "dev" {
  inherits = ["_main-common"]
  target   = "dev"
  tags     = ["tt-metalium-dev:local"]
}

target "release" {
  inherits = ["_main-common"]
  target   = "release"
  tags     = ["tt-metalium-release:local"]
}

target "release-models" {
  inherits = ["_main-common"]
  target   = "release-models"
  tags     = ["tt-metalium-release-models:local"]
}

group "main" {
  targets = ["ci-build", "ci-test", "dev"]
}

# =============================================================================
# Basic image targets (from Dockerfile.basic-dev)
#
# Uses a subset of tools: cmake, sfpi, openmpi, ccache
# =============================================================================

target "_basic-common" {
  context    = "."
  dockerfile = "dockerfile/Dockerfile.basic-dev"
  args = {
    UBUNTU_VERSION = UBUNTU_VERSION
    PYTHON_VERSION = PYTHON_VERSION
  }
  contexts = {
    cmake-layer   = "target:cmake"
    sfpi-layer    = "target:sfpi"
    openmpi-layer = "target:openmpi"
    ccache-layer  = "target:ccache"
  }
}

target "basic-dev" {
  inherits = ["_basic-common"]
  target   = "base"
  tags     = ["tt-metalium-basic-dev:local"]
}

target "basic-ttnn-runtime" {
  inherits = ["_basic-common"]
  target   = "basic-ttnn-runtime"
  tags     = ["tt-metalium-basic-ttnn-runtime:local"]
}

group "basic" {
  targets = ["basic-dev", "basic-ttnn-runtime"]
}

# =============================================================================
# ManyLinux target (from Dockerfile.manylinux)
#
# Uses a subset of tools: ccache, mold, sfpi, openmpi
# =============================================================================

target "manylinux" {
  context    = "."
  dockerfile = "dockerfile/Dockerfile.manylinux"
  tags       = ["tt-metalium-manylinux:local"]
  contexts = {
    ccache-layer  = "target:ccache"
    mold-layer    = "target:mold"
    sfpi-layer    = "target:sfpi"
    openmpi-layer = "target:openmpi"
  }
}

# =============================================================================
# Evaluation target (from Dockerfile.evaluation)
#
# Uses ccache and sfpi only.
# =============================================================================

target "evaluation" {
  context    = "."
  dockerfile = "dockerfile/Dockerfile.evaluation"
  tags       = ["tt-metalium-evaluation:local"]
  contexts = {
    ccache-layer = "target:ccache"
    sfpi-layer   = "target:sfpi"
  }
}

# =============================================================================
# Top-level groups
# =============================================================================

group "default" {
  targets = ["dev"]
}

group "all" {
  targets = ["tools", "venvs", "main", "basic", "manylinux", "evaluation"]
}
