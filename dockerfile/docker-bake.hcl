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
#   docker buildx bake dev-light        # Build dev image without venv layers
#   docker buildx bake ci-test          # Build CI test image
#   docker buildx bake ci-build         # Build CI build image
#   docker buildx bake basic-dev        # Build basic dev image
#   docker buildx bake manylinux        # Build manylinux image
#   docker buildx bake tools            # Build just the tool images
#   docker buildx bake venvs            # Build just the venv images
#   docker buildx bake all              # Build everything
#   docker buildx bake --print dev      # Dry run: show what would be built
#   docker buildx bake --print dev-light
#
#   # Ubuntu 24.04 variant (PYTHON_VERSION must match — 22.04→3.10, 24.04→3.12):
#   UBUNTU_VERSION=24.04 PYTHON_VERSION=3.12 docker buildx bake dev
#
#   # Force rebuild (no cache):
#   docker buildx bake --no-cache dev
#
# USAGE (CI / GHCR overrides):
#   Workflows override tool/venv contexts to point to pre-built GHCR images and
#   invoke bake via .github/actions/manual-docker-bake (manual CLI instead of
#   docker/bake-action, which had registry metadata timeout issues).
#     docker buildx bake ci-build ci-test dev-light dev \
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
# The `contexts` feature wires required tool/venv outputs into downstream builds:
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

variable "TT_SMI_VERSION" {
  # As of June 2026: this is where you set SMI version of Metal container image.
  # Bake always passes this to the main targets, so it takes precedence over the
  # ARG TT_SMI_VERSION default in dockerfile/Dockerfile (used only for standalone
  # `docker build` without Bake). Keep the two in sync.
  default = "5.2.0"
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

target "clangbuildanalyzer" {
  context    = "."
  dockerfile = "dockerfile/Dockerfile.tools"
  target     = "clangbuildanalyzer"
  tags       = ["tool-clangbuildanalyzer:local"]
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

target "zstd" {
  context    = "."
  dockerfile = "dockerfile/Dockerfile.tools"
  target     = "zstd"
  tags       = ["tool-zstd:local"]
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
  targets = ["ccache", "clangbuildanalyzer", "cmake", "doxygen", "gdb", "mold", "openmpi", "sfpi", "yq", "zstd"]
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
# Use Bake's `contexts` to wire in tool layers for all main targets and venv
# layers only for targets that copy them. Bake builds referenced targets
# automatically before the main image.
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
    TT_SMI_VERSION = TT_SMI_VERSION
  }
  contexts = {
    # Tool layers (resolved from Dockerfile.tools targets locally)
    ccache-layer             = "target:ccache"
    mold-layer               = "target:mold"
    doxygen-layer            = "target:doxygen"
    clangbuildanalyzer-layer = "target:clangbuildanalyzer"
    gdb-layer                = "target:gdb"
    cmake-layer              = "target:cmake"
    yq-layer                 = "target:yq"
    zstd-layer               = "target:zstd"
    sfpi-layer               = "target:sfpi"
    openmpi-layer            = "target:openmpi"
  }
}

target "ci-build-light" {
  inherits = ["_main-common"]
  target   = "ci-build-light"
  tags     = ["tt-metalium-ci-build-light:local"]
}

target "ci-build" {
  inherits = ["_main-common"]
  target   = "ci-build"
  tags     = ["tt-metalium-ci-build:local"]
  contexts = {
    ci-build-venv-layer = "target:ci-build-venv"
  }
}

target "ci-test-light" {
  inherits = ["_main-common"]
  target   = "ci-test-light"
  tags     = ["tt-metalium-ci-test-light:local"]
}

target "ci-test" {
  inherits = ["_main-common"]
  target   = "ci-test"
  tags     = ["tt-metalium-ci-test:local"]
  contexts = {
    ci-test-venv-layer = "target:ci-test-venv"
  }
}

target "dev-light" {
  inherits = ["_main-common"]
  target   = "dev-light"
  tags     = ["tt-metalium-dev-light:local"]
}

target "dev" {
  inherits = ["_main-common"]
  target   = "dev"
  tags     = ["tt-metalium-dev:local"]
  contexts = {
    ci-test-venv-layer = "target:ci-test-venv"
  }
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
  targets = ["ci-build-light", "ci-build", "ci-test-light", "ci-test", "dev-light", "dev"]
}

# =============================================================================
# Basic image targets (from Dockerfile.basic-dev)
#
# Uses a subset of tools: cmake, openmpi, ccache (SFPI is only in basic-ttnn-runtime, not basic-dev)
# =============================================================================

target "_basic-common" {
  context    = "."
  dockerfile = "dockerfile/Dockerfile.basic-dev"
  args = {
    UBUNTU_VERSION = UBUNTU_VERSION
    PYTHON_VERSION = PYTHON_VERSION
    UV_IMAGE       = UV_IMAGE
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
    zstd-layer    = "target:zstd"
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
  args = {
    UV_IMAGE = UV_IMAGE
  }
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
