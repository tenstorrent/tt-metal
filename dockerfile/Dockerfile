# TT-Metalium Dockerfile
# Optimized with multi-stage builds for tool and Python dependency caching
#
# Third-party tools are pulled from pre-built GHCR images to avoid hitting
# upstream endpoints repeatedly. Tool images are built by Dockerfile.tools
# and pushed to GHCR by build-docker-artifact.yaml.
#
# Python virtual environments with all dependencies pre-installed are pulled
# from pre-built GHCR images to avoid repeated pip install operations.
# Python venv images are built by Dockerfile.python and pushed to GHCR.
#
# =============================================================================
# LOCAL BUILD INSTRUCTIONS (without GHCR)
# =============================================================================
#
# Step 1: Build all tool images
#
#   docker build -f dockerfile/Dockerfile.tools --target ccache -t tool-ccache:local .
#   docker build -f dockerfile/Dockerfile.tools --target mold -t tool-mold:local .
#   docker build -f dockerfile/Dockerfile.tools --target doxygen -t tool-doxygen:local .
#   docker build -f dockerfile/Dockerfile.tools --target cba -t tool-cba:local .
#   docker build -f dockerfile/Dockerfile.tools --target gdb -t tool-gdb:local .
#   docker build -f dockerfile/Dockerfile.tools --target cmake -t tool-cmake:local .
#   docker build -f dockerfile/Dockerfile.tools --target yq -t tool-yq:local .
#   docker build -f dockerfile/Dockerfile.tools --target sfpi -t tool-sfpi:local .
#   docker build -f dockerfile/Dockerfile.tools --target openmpi -t tool-openmpi:local .
#
# Step 2: Build Python venv images
#
#   docker build -f dockerfile/Dockerfile.python --target ci-build-venv \
#     --build-arg UBUNTU_VERSION=22.04 -t python-ci-build-venv:local .
#   docker build -f dockerfile/Dockerfile.python --target ci-test-venv \
#     --build-arg UBUNTU_VERSION=22.04 -t python-ci-test-venv:local .
#
# Step 3: Build the main image (e.g., dev target)
#
#   docker build -f dockerfile/Dockerfile --target dev \
#     --build-arg UBUNTU_VERSION=22.04 \
#     --build-arg TOOL_CCACHE_IMAGE=tool-ccache:local \
#     --build-arg TOOL_MOLD_IMAGE=tool-mold:local \
#     --build-arg TOOL_DOXYGEN_IMAGE=tool-doxygen:local \
#     --build-arg TOOL_CBA_IMAGE=tool-cba:local \
#     --build-arg TOOL_GDB_IMAGE=tool-gdb:local \
#     --build-arg TOOL_CMAKE_IMAGE=tool-cmake:local \
#     --build-arg TOOL_YQ_IMAGE=tool-yq:local \
#     --build-arg TOOL_SFPI_IMAGE=tool-sfpi:local \
#     --build-arg TOOL_OPENMPI_IMAGE=tool-openmpi:local \
#     --build-arg PYTHON_CI_BUILD_VENV_IMAGE=python-ci-build-venv:local \
#     --build-arg PYTHON_CI_TEST_VENV_IMAGE=python-ci-test-venv:local \
#     -t tt-metalium-dev:local .
#
# Available targets: ci-build, ci-test, dev, release, release-models
# =============================================================================

#############################################################
# Tool image configuration (pre-built images from GHCR)
# NOTE: ARGs used in FROM must be declared before any FROM statement
#############################################################

# Ubuntu base image version
ARG UBUNTU_VERSION=22.04

# Tool images from GHCR (built by Dockerfile.tools, pushed by build-docker-artifact.yaml)
# These default to placeholder values; the CI workflow passes actual GHCR tags.
# For local builds, either build the tool images first or let Docker build from scratch.
ARG TOOL_CCACHE_IMAGE=scratch
ARG TOOL_MOLD_IMAGE=scratch
ARG TOOL_DOXYGEN_IMAGE=scratch
ARG TOOL_CBA_IMAGE=scratch
ARG TOOL_GDB_IMAGE=scratch
ARG TOOL_CMAKE_IMAGE=scratch
ARG TOOL_YQ_IMAGE=scratch
ARG TOOL_SFPI_IMAGE=scratch
ARG TOOL_OPENMPI_IMAGE=scratch

# Python venv images from GHCR (built by Dockerfile.python, pushed by build-docker-artifact.yaml)
# These contain pre-built virtual environments with all Python dependencies installed.
# Using pre-built venvs saves 5-10 minutes per build by avoiding pip install operations.
# Set to scratch to fall back to building venv in-place (slower but works without GHCR).
ARG PYTHON_CI_BUILD_VENV_IMAGE=scratch
ARG PYTHON_CI_TEST_VENV_IMAGE=scratch

# uv - Python package manager
# Pinned to specific SHA256 for reproducible builds per:
# https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
ARG UV_VERSION=0.9.26
ARG UV_IMAGE_DIGEST=sha256:9a23023be68b2ed09750ae636228e903a54a05ea56ed03a934d00fe9fbeded4b

#############################################################
# Tool layers - pulled from pre-built GHCR images
# These images contain binaries at /install/{bin,lib,...}
#############################################################

# ccache layer - pre-built from GHCR
FROM ${TOOL_CCACHE_IMAGE} AS ccache-layer

# mold layer - pre-built from GHCR
FROM ${TOOL_MOLD_IMAGE} AS mold-layer

# doxygen layer - pre-built from GHCR
FROM ${TOOL_DOXYGEN_IMAGE} AS doxygen-layer

# ClangBuildAnalyzer layer - pre-built from GHCR
FROM ${TOOL_CBA_IMAGE} AS cba-layer

# GDB layer - pre-built from GHCR
FROM ${TOOL_GDB_IMAGE} AS gdb-layer

# cmake layer - pre-built from GHCR
FROM ${TOOL_CMAKE_IMAGE} AS cmake-layer

# yq layer - pre-built from GHCR
FROM ${TOOL_YQ_IMAGE} AS yq-layer

# SFPI layer - pre-built from GHCR
FROM ${TOOL_SFPI_IMAGE} AS sfpi-layer

# OpenMPI layer - pre-built from GHCR (ULFM support)
FROM ${TOOL_OPENMPI_IMAGE} AS openmpi-layer

# Python venv layers - pre-built from GHCR
# These contain complete virtual environments with all dependencies pre-installed
FROM ${PYTHON_CI_BUILD_VENV_IMAGE} AS ci-build-venv-layer
FROM ${PYTHON_CI_TEST_VENV_IMAGE} AS ci-test-venv-layer

#############################################################
# uv layer - copy from official distroless image
# Using SHA256 pinning for reproducible builds
# See: https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
#############################################################

FROM ghcr.io/astral-sh/uv@sha256:9a23023be68b2ed09750ae636228e903a54a05ea56ed03a934d00fe9fbeded4b AS uv-layer

#############################################################
# Base Ubuntu image with system dependencies
#############################################################

FROM mirror.gcr.io/ubuntu:${UBUNTU_VERSION} AS base

ENV DEBIAN_FRONTEND=noninteractive

# OpenMPI configuration - must match Dockerfile.tools OMPI_TAG
# The openmpi-layer provides /install/openmpi-<tag>-ulfm/ which we copy to /opt/
ARG OMPI_TAG=v5.0.7
ENV OMPI_PREFIX=/opt/openmpi-${OMPI_TAG}-ulfm
ENV PATH=${OMPI_PREFIX}/bin:$PATH
ENV LD_LIBRARY_PATH=${OMPI_PREFIX}/lib
ENV CPATH=${OMPI_PREFIX}/include
ENV PKG_CONFIG_PATH=${OMPI_PREFIX}/lib/pkgconfig

COPY --from=uv-layer /uv /uvx /usr/local/bin/
COPY --from=openmpi-layer /install/ /opt/
COPY --from=cmake-layer /install/ /usr/local/

# Install runtime deps (SFPI already copied from tool image above)
COPY /install_dependencies.sh /opt/tt_metal_infra/scripts/docker/install_dependencies.sh
COPY /tt_metal/sfpi-info.sh /opt/tt_metal_infra/scripts/docker/sfpi-info.sh
COPY /tt_metal/sfpi-version /opt/tt_metal_infra/scripts/docker/sfpi-version
RUN /bin/bash /opt/tt_metal_infra/scripts/docker/install_dependencies.sh --docker

# Copy SFPI from pre-built tool image
RUN --mount=from=sfpi-layer,source=/install,target=/sfpi-staging \
    mkdir -p /opt/tenstorrent && \
    cp -a /sfpi-staging/opt/tenstorrent/sfpi /opt/tenstorrent/

#############################################################
# CI Build image
#############################################################

FROM base AS ci-build

# Ensure /usr/local/bin is in PATH so CMake can find mold (ld.mold)
ENV PATH="/usr/local/bin:$PATH"
ENV PYTHON_ENV_DIR=/opt/venv
ENV PATH="$PYTHON_ENV_DIR/bin:$PATH"
ENV VIRTUAL_ENV="$PYTHON_ENV_DIR"
ENV CCACHE_TEMPDIR=/tmp/ccache

# Copy pre-built tools from GHCR tool images
# These images are built by Dockerfile.tools and pushed by build-docker-artifact.yaml
COPY --from=ccache-layer /install/bin/ccache /usr/local/bin/
COPY --from=doxygen-layer /install/ /usr/local/
COPY --from=cba-layer /install/ /usr/local/
COPY --from=yq-layer /install/bin/yq /usr/local/bin/

# Copy pre-built mold from GHCR tool image
COPY --from=mold-layer /install/bin/ /usr/local/bin/
COPY --from=mold-layer /install/lib/ /usr/local/lib/

# Copy pre-built ci-build venv from tool image
# Set umask 000 to allow all users to write (required for uv pip install at runtime)
# SECURITY NOTE: This enables any user in the container to modify the Python environment.
RUN --mount=from=ci-build-venv-layer,source=/install,target=/venv-staging \
    umask 000 && \
    cp -a /venv-staging/venv $PYTHON_ENV_DIR && \
    umask 002 && \
    echo "source $PYTHON_ENV_DIR/bin/activate" >> /etc/bash.bashrc

RUN git config --global --add safe.directory '*'

# Install extra ci apt requirements
# Note: llvm-20 is required for llvm-ar-20, which CMake needs for LTO support detection
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    bc \
    libclang-20-dev \
    libclang-rt-20-dev \
    clang-tidy-20 \
    clang-20 \
    llvm-20 \
    curl \
    dialog \
    file \
    graphviz \
    jq \
    sudo \
    uuid-runtime \
    wget \
    zstd \
    libgtest-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


#############################################################
# CI Test image
#############################################################

FROM ci-build AS ci-test

ARG UBUNTU_VERSION=22.04
ARG TT_METAL_INFRA_DIR=/opt/tt_metal_infra

# Create directories for infra (needed even with pre-built venv for requirements file reference)
RUN mkdir -p ${TT_METAL_INFRA_DIR}/tt-metal/docs/ && \
    mkdir -p ${TT_METAL_INFRA_DIR}/tt-metal/tests/sweep_framework/ && \
    mkdir -p ${TT_METAL_INFRA_DIR}/tt-metal/tt_metal/python_env/ && \
    mkdir -p ${TT_METAL_INFRA_DIR}/tools/triage/

# Copy requirements files (needed for fallback install if no pre-built venv)
COPY /docs/requirements-docs.txt ${TT_METAL_INFRA_DIR}/tt-metal/docs/.
COPY /tests/sweep_framework/requirements-sweeps.txt ${TT_METAL_INFRA_DIR}/tt-metal/tests/sweep_framework/.
COPY /tt_metal/python_env/requirements-dev.txt ${TT_METAL_INFRA_DIR}/tt-metal/tt_metal/python_env/.
COPY /tools/triage/requirements.txt ${TT_METAL_INFRA_DIR}/tools/triage/requirements.txt

# Copy pre-built ci-test venv from tool image
RUN --mount=from=ci-test-venv-layer,source=/install,target=/venv-staging \
    umask 000 && \
    rm -rf $PYTHON_ENV_DIR && \
    cp -a /venv-staging/venv $PYTHON_ENV_DIR

COPY /scripts/install_debugger.sh ${TT_METAL_INFRA_DIR}/scripts/install_debugger.sh
COPY /scripts/ttexalens_ref.txt ${TT_METAL_INFRA_DIR}/scripts/ttexalens_ref.txt
RUN umask 000 && \
    if [ "$UBUNTU_VERSION" = "22.04" ]; then \
    echo "Installing debugger for Ubuntu 22.04..." && \
    ${TT_METAL_INFRA_DIR}/scripts/install_debugger.sh && \
    echo "Debugger installation completed" && \
    python3 -c "import ttexalens; print('ttexalens successfully imported')" || echo "ttexalens import failed"; \
    fi

#############################################################
# Development image
#############################################################

FROM ci-test AS dev

ARG UBUNTU_VERSION=22.04

COPY /dockerfile/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

RUN --mount=from=gdb-layer,source=/install,target=/gdb-staging \
    if [ "$UBUNTU_VERSION" != "24.04" ]; then \
        cp -a /gdb-staging/* /usr/local/; \
    else \
        apt-get update && apt-get install -y --no-install-recommends gdb && \
        apt-get clean && rm -rf /var/lib/apt/lists/*; \
    fi

# Install dev deps
# libgl1-mesa-glx is needed for Yolo models https://github.com/tenstorrent/tt-metal/pull/19899
# libgl1-mesa-glx does not exist in ubuntu 24.04 which is right around the corner
# switching to libgl1
RUN apt-get update && apt-get install -y --no-install-recommends \
    acl \
    emacs \
    less \
    libmpfr-dev \
    nano \
    openssh-server \
    vim \
    libgl1 \
    ipmitool \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# IWYU could be useful to developers
# Not currently building, and unclear whether this is used; can put it back in if needed
# RUN mkdir -p /tmp/iwyu \
#     && wget -O /tmp/iwyu/iwyu.tar.gz https://github.com/include-what-you-use/include-what-you-use/archive/refs/tags/0.21.tar.gz \
#     && tar -xzf /tmp/iwyu/iwyu.tar.gz -C /tmp/iwyu --strip-components=1 \
#     && cmake -S /tmp/iwyu/ -B /tmp/iwyu/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-20 -DCMAKE_CXX_COMPILER=clang++-20 \
#     && cmake --build /tmp/iwyu/build --parallel \
#     && cmake --install /tmp/iwyu/build \
#     && rm -rf /tmp/iwyu

#############################################################
# Release image
#############################################################

FROM base AS release

# Set up virtual environment
ENV PYTHON_ENV_DIR=/opt/venv
ENV PATH="$PYTHON_ENV_DIR/bin:$PATH"
ENV VIRTUAL_ENV="$PYTHON_ENV_DIR"
# Set umask 000 to allow all users to write (required for uv pip install at runtime)
# SECURITY NOTE: This enables any user in the container to modify the Python environment.
# This is acceptable for single-user containers but should be reconsidered for multi-tenant
# or security-sensitive deployments. Consider using user-specific venvs in those cases.
RUN umask 000 && uv venv --relocatable $PYTHON_ENV_DIR \
    && umask 022 \
    && "source $PYTHON_ENV_DIR/bin/activate" >> /etc/bash.bashrc \
    && mkdir -p /etc \
    && echo "[global]\nextra-index-url = https://download.pytorch.org/whl/cpu" > /etc/pip.conf

ARG WHEEL_FILENAME
ADD $WHEEL_FILENAME $WHEEL_FILENAME
RUN uv pip install $WHEEL_FILENAME

#############################################################
# Release models image
#############################################################

FROM base AS release-models

# Set up virtual environment and TT_METAL_HOME
ENV PYTHON_ENV_DIR=/opt/venv
ENV TT_METAL_HOME=/tt-metal
ENV PYTHONPATH=/tt-metal
ENV PATH="$PYTHON_ENV_DIR/bin:$PATH"
ENV VIRTUAL_ENV="$PYTHON_ENV_DIR"

# Accept git reference argument
ARG GIT_REF=main

# Clone tt-metal
RUN /bin/bash -c "git clone --filter=blob:none --recurse-submodules --tags \
    https://github.com/tenstorrent/tt-metal.git ${TT_METAL_HOME} \
    && cd ${TT_METAL_HOME} \
    && git checkout ${GIT_REF} \
    && git submodule update --init --recursive"

WORKDIR /tt-metal

# Build tt-metal and venv, then clear cache
# Set umask 000 to allow all users to write (required for uv pip install at runtime)
# SECURITY NOTE: This enables any user in the container to modify the Python environment.
# This is acceptable for single-user containers but should be reconsidered for multi-tenant
# or security-sensitive deployments. Consider using user-specific venvs in those cases.
RUN umask 000 && /bin/bash -c "./build_metal.sh \
    && ./create_venv.sh \
    && source $PYTHON_ENV_DIR/bin/activate \
    && rm -rf /root/.cache /root/.cargo /tt-metal/.cpmcache" \
    && umask 022 \
    && echo "source $PYTHON_ENV_DIR/bin/activate" >> /etc/bash.bashrc

# Extra system deps
RUN apt-get update && apt-get install -y \
    # required
    gosu \
    libgl1 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
