# TT-Metalium Dockerfile
# Optimized with multi-stage builds for tool caching

#############################################################
# Tool version and hash configuration (centralized)
# NOTE: ARGs used in FROM must be declared before any FROM statement
#############################################################

# Ubuntu base image version
ARG UBUNTU_VERSION=22.04

# uv - Python package manager
# Pinned to specific SHA256 for reproducible builds per:
# https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
ARG UV_VERSION=0.9.26
ARG UV_IMAGE_DIGEST=sha256:9a23023be68b2ed09750ae636228e903a54a05ea56ed03a934d00fe9fbeded4b

ARG CCACHE_VERSION=4.10.2
ARG CCACHE_SHA256=80cab87bd510eca796467aee8e663c398239e0df1c4800a0b5dff11dca0b4f18

ARG MOLD_VERSION=2.40.4
ARG MOLD_SHA256=4c999e19ffa31afa5aa429c679b665d5e2ca5a6b6832ad4b79668e8dcf3d8ec1

ARG DOXYGEN_VERSION=1.16.1
ARG DOXYGEN_SHA256=a56f885d37e3aae08a99f638d17bbb381224c03a878d9e2dda4f9fa4baf1d8bd

ARG CBA_VERSION=1.6.0
ARG CBA_SHA256=868a8d34ecb9b65da4e5874342062a12c081ce4385c7ddd6ce7d557a0c5c292d

ARG GDB_VERSION=14.2
ARG GDB_SHA256=2d4dd8061d8ded12b6c63f55e45344881e8226105f4d2a9b234040efa5ce7772

#############################################################
# ccache layer - simple binary extraction (no dependencies)
#############################################################

FROM alpine:3.19 AS ccache-layer
ARG CCACHE_VERSION
ARG CCACHE_SHA256
COPY dockerfile/scripts/install-ccache.sh /tmp/install-ccache.sh
RUN apk add --no-cache bash wget && \
    CCACHE_VERSION=${CCACHE_VERSION} \
    CCACHE_SHA256=${CCACHE_SHA256} \
    INSTALL_DIR=/install \
    /bin/bash /tmp/install-ccache.sh

#############################################################
# mold layer - simple binary extraction (no dependencies)
#############################################################

FROM alpine:3.19 AS mold-layer
ARG MOLD_VERSION
ARG MOLD_SHA256
COPY dockerfile/scripts/install-mold.sh /tmp/install-mold.sh
RUN apk add --no-cache bash wget && \
    MOLD_VERSION=${MOLD_VERSION} \
    MOLD_SHA256=${MOLD_SHA256} \
    INSTALL_DIR=/install \
    /bin/bash /tmp/install-mold.sh

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

# Install runtime deps
COPY /install_dependencies.sh /opt/tt_metal_infra/scripts/docker/install_dependencies.sh
COPY /tt_metal/sfpi-info.sh /opt/tt_metal_infra/scripts/docker/sfpi-info.sh
COPY /tt_metal/sfpi-version /opt/tt_metal_infra/scripts/docker/sfpi-version
RUN /bin/bash /opt/tt_metal_infra/scripts/docker/install_dependencies.sh --docker

# Install uv from official distroless image (pinned by SHA256)
COPY --from=uv-layer /uv /uvx /usr/local/bin/

#############################################################
# doxygen builder - requires make to install
#############################################################

FROM base AS doxygen-builder
ARG DOXYGEN_VERSION
ARG DOXYGEN_SHA256

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget make \
    && rm -rf /var/lib/apt/lists/*

COPY dockerfile/scripts/install-doxygen.sh /tmp/install-doxygen.sh
RUN DOXYGEN_VERSION=${DOXYGEN_VERSION} \
    DOXYGEN_SHA256=${DOXYGEN_SHA256} \
    INSTALL_PREFIX=/staging \
    /bin/bash /tmp/install-doxygen.sh

#############################################################
# ClangBuildAnalyzer builder - requires cmake to build
#############################################################

FROM base AS cba-builder
ARG CBA_VERSION
ARG CBA_SHA256

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY dockerfile/scripts/install-clangbuildanalyzer.sh /tmp/install-clangbuildanalyzer.sh
RUN CBA_VERSION=${CBA_VERSION} \
    CBA_SHA256=${CBA_SHA256} \
    INSTALL_PREFIX=/staging \
    /bin/bash /tmp/install-clangbuildanalyzer.sh

#############################################################
# GDB builder - requires configure/make to build
#############################################################

FROM base AS gdb-builder
ARG GDB_VERSION
ARG GDB_SHA256

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget build-essential texinfo libgmp-dev libmpfr-dev \
    && rm -rf /var/lib/apt/lists/*

COPY dockerfile/scripts/install-gdb.sh /tmp/install-gdb.sh
RUN GDB_VERSION=${GDB_VERSION} \
    GDB_SHA256=${GDB_SHA256} \
    INSTALL_PREFIX=/staging \
    /bin/bash /tmp/install-gdb.sh

#############################################################
# CI Build image
#############################################################

FROM base AS ci-build

# Copy pre-built ccache from layer (no redownload/rebuild on base changes)
COPY --from=ccache-layer /install/bin/ccache /usr/local/bin/

# Copy pre-built doxygen from builder stage
COPY --from=doxygen-builder /staging/ /usr/local/

# Copy pre-built ClangBuildAnalyzer from builder stage
COPY --from=cba-builder /staging/ /usr/local/

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

# Install yq (YAML processor) for CI workflows
# Note: mikefarah/yq is not available in apt, download binary from GitHub releases
ARG YQ_VERSION=v4.44.6
RUN wget -q "https://github.com/mikefarah/yq/releases/download/${YQ_VERSION}/yq_linux_amd64" -O /usr/local/bin/yq && \
    chmod +x /usr/local/bin/yq

# Set up virtual environment using uv
# Uses the system Python version (3.10 on Ubuntu 22.04, 3.12 on Ubuntu 24.04)

ENV PYTHON_ENV_DIR=/opt/venv
# Set umask 000 to allow all users to write (required for uv pip install at runtime)
# SECURITY NOTE: This enables any user in the container to modify the Python environment.
# This is acceptable for single-user containers but should be reconsidered for multi-tenant
# or security-sensitive deployments. Consider using user-specific venvs in those cases.
RUN umask 000 && uv venv --relocatable $PYTHON_ENV_DIR

# Ensure the virtual environment is used for all Python-related commands
ENV PATH="$PYTHON_ENV_DIR/bin:$PATH"
ENV VIRTUAL_ENV="$PYTHON_ENV_DIR"

# Ensure the virtual environment is activated on shell startup
RUN echo "source $PYTHON_ENV_DIR/bin/activate" >> /etc/bash.bashrc

# Numpy needed for tt-train to build
RUN umask 000 && uv pip install --no-cache-dir build numpy setuptools

RUN git config --global --add safe.directory '*'

ENV CCACHE_TEMPDIR=/tmp/ccache

# Copy pre-built mold from layer (no redownload/rebuild on base changes)
COPY --from=mold-layer /install/bin/ /usr/local/bin/
COPY --from=mold-layer /install/lib/ /usr/local/lib/

# Ensure /usr/local/bin is in PATH so CMake can find mold (ld.mold)
ENV PATH="/usr/local/bin:$PATH"

#############################################################
# CI Test image
#############################################################

FROM ci-build AS ci-test

ARG UBUNTU_VERSION=22.04
ARG TT_METAL_INFRA_DIR=/opt/tt_metal_infra

# Create directories for infra
RUN mkdir -p ${TT_METAL_INFRA_DIR}/tt-metal/docs/
RUN mkdir -p ${TT_METAL_INFRA_DIR}/tt-metal/tests/sweep_framework/
RUN mkdir -p ${TT_METAL_INFRA_DIR}/tt-metal/tt_metal/python_env/
# Copy requirements from tt-metal folders with requirements.txt docs
COPY /docs/requirements-docs.txt ${TT_METAL_INFRA_DIR}/tt-metal/docs/.
# Copy requirements from tt-metal folders for sweeps (requirements-sweeps.txt)
COPY /tests/sweep_framework/requirements-sweeps.txt ${TT_METAL_INFRA_DIR}/tt-metal/tests/sweep_framework/.
COPY /tt_metal/python_env/requirements-dev.txt ${TT_METAL_INFRA_DIR}/tt-metal/tt_metal/python_env/.
COPY /tools/triage/requirements.txt ${TT_METAL_INFRA_DIR}/tools/triage/requirements.txt

# Configure PyTorch CPU index and install Python packages using uv
# index-strategy unsafe-best-match is used to avoid pulling in transitive dependencies
# that are not explicitly listed in the requirements.txt files
RUN umask 000 && \
    uv pip install --no-cache --index-url https://download.pytorch.org/whl/cpu torch && \
    uv pip install --index-strategy unsafe-best-match --no-cache -r ${TT_METAL_INFRA_DIR}/tt-metal/tt_metal/python_env/requirements-dev.txt && \
    uv pip install --index-strategy unsafe-best-match --no-cache -r ${TT_METAL_INFRA_DIR}/tt-metal/docs/requirements-docs.txt && \
    uv pip install --index-strategy unsafe-best-match --no-cache -r ${TT_METAL_INFRA_DIR}/tools/triage/requirements.txt

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

# Copy pre-built GDB for non-24.04 Ubuntu versions
# 24.04 has gdb 15.1 by default which is sufficient
RUN set -eux; echo "Ubuntu version: $UBUNTU_VERSION"
RUN --mount=from=gdb-builder,source=/staging,target=/gdb-staging \
    if [ "$UBUNTU_VERSION" != "24.04" ]; then \
        echo "Installing custom GDB for Ubuntu $UBUNTU_VERSION" && \
        cp -a /gdb-staging/* /usr/local/; \
    else \
        echo "Using system GDB (15.1) for Ubuntu 24.04 - no custom build needed" && \
        apt-get update && apt-get install -y --no-install-recommends gdb && \
        apt-get clean && rm -rf /var/lib/apt/lists/*; \
    fi

# IWYU could be useful to developers
# Not currently building, and unclear whether this is used; can put it back in if needed
# RUN mkdir -p /tmp/iwyu \
#     && wget -O /tmp/iwyu/iwyu.tar.gz https://github.com/include-what-you-use/include-what-you-use/archive/refs/tags/0.21.tar.gz \
#     && tar -xzf /tmp/iwyu/iwyu.tar.gz -C /tmp/iwyu --strip-components=1 \
#     && cmake -S /tmp/iwyu/ -B /tmp/iwyu/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-20 -DCMAKE_CXX_COMPILER=clang++-20 \
#     && cmake --build /tmp/iwyu/build --parallel \
#     && cmake --install /tmp/iwyu/build \
#     && rm -rf /tmp/iwyu

COPY /dockerfile/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

#############################################################
# Release image
#############################################################

FROM base AS release

# Set up virtual environment
ENV PYTHON_ENV_DIR=/opt/venv
# Set umask 000 to allow all users to write (required for uv pip install at runtime)
# SECURITY NOTE: This enables any user in the container to modify the Python environment.
# This is acceptable for single-user containers but should be reconsidered for multi-tenant
# or security-sensitive deployments. Consider using user-specific venvs in those cases.
RUN umask 000 && uv venv --relocatable $PYTHON_ENV_DIR

# Ensure the virtual environment is used for all Python-related commands
ENV PATH="$PYTHON_ENV_DIR/bin:$PATH"
ENV VIRTUAL_ENV="$PYTHON_ENV_DIR"

# Ensure the virtual environment is activated on shell startup
RUN echo "source $PYTHON_ENV_DIR/bin/activate" >> /etc/bash.bashrc

RUN mkdir -p /etc && \
    echo "[global]\nextra-index-url = https://download.pytorch.org/whl/cpu" > /etc/pip.conf

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
    && rm -rf /root/.cache /root/.cargo /tt-metal/.cpmcache"

# Ensure the virtual environment is used for all Python-related commands
ENV PATH="$PYTHON_ENV_DIR/bin:$PATH"
ENV VIRTUAL_ENV="$PYTHON_ENV_DIR"

# Ensure the virtual environment is activated on shell startup
RUN echo "source $PYTHON_ENV_DIR/bin/activate" >> /etc/bash.bashrc

# Extra system deps
RUN apt-get update && apt-get install -y \
    # required
    gosu \
    libgl1 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
