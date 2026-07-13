# TT-Metalium Dockerfile
# Optimized with multi-stage builds for tool and Python dependency caching.
#
# Tool and venv layers are injected via Docker Buildx Bake contexts
# (see docker-bake.hcl). The FROM scratch stubs below are overridden by
# Bake with either local tool target outputs or pre-built GHCR images.
#
# =============================================================================
# BUILD INSTRUCTIONS
# =============================================================================
#
#   docker buildx bake -f dockerfile/docker-bake.hcl dev
#   docker buildx bake -f dockerfile/docker-bake.hcl dev-light
#   docker buildx bake -f dockerfile/docker-bake.hcl ci-test
#   docker buildx bake -f dockerfile/docker-bake.hcl ci-build-light
#   docker buildx bake -f dockerfile/docker-bake.hcl ci-build
#   docker buildx bake -f dockerfile/docker-bake.hcl ci-test-light
#   UBUNTU_VERSION=24.04 PYTHON_VERSION=3.12 docker buildx bake -f dockerfile/docker-bake.hcl dev
#   docker buildx bake -f dockerfile/docker-bake.hcl --print dev   # dry run
#
# Or use the wrapper script:
#   ./dockerfile/build-local.sh dev
#   ./dockerfile/build-local.sh --help
#
# Available targets: ci-build-light, ci-build, ci-test-light, ci-test, dev-light, dev, release, release-models
# =============================================================================

#############################################################
# Build arguments
#############################################################

ARG UBUNTU_VERSION=22.04
# As of June 2026: this is where you set SMI version of Metal container image
# for CI and things like IRD or other workflows using the dev container.
ARG TT_SMI_VERSION=5.2.0
# PYTHON_VERSION can be explicitly set or auto-detected from UBUNTU_VERSION:
# Ubuntu 22.04 -> Python 3.10
# Ubuntu 24.04 -> Python 3.12
ARG PYTHON_VERSION=""
# Fallback UV_IMAGE for standalone `docker build` without Bake.
# The single source of truth is docker-bake.hcl variable "UV_IMAGE".
# When upgrading uv, update docker-bake.hcl first, then sync this default.
ARG UV_IMAGE=ghcr.io/astral-sh/uv@sha256:9a23023be68b2ed09750ae636228e903a54a05ea56ed03a934d00fe9fbeded4b

#############################################################
# Tool and venv layers (resolved by docker-bake.hcl contexts)
#
# These FROM scratch stubs are overridden by Bake:
#   - Locally: Bake builds Dockerfile.tools/Dockerfile.python targets first
#   - In CI:   Bake substitutes pre-built GHCR images via --set contexts
# WARNING: Plain `docker build` is NOT supported — it will fail because these
# stubs resolve to empty scratch images. Always use `docker buildx bake`.
#############################################################

FROM scratch AS ccache-layer
FROM scratch AS mold-layer
FROM scratch AS doxygen-layer
FROM scratch AS clangbuildanalyzer-layer
FROM scratch AS gdb-layer
FROM scratch AS cmake-layer
FROM scratch AS yq-layer
FROM scratch AS zstd-layer
FROM scratch AS sfpi-layer
FROM scratch AS openmpi-layer
FROM scratch AS ci-build-venv-layer
FROM scratch AS ci-test-venv-layer

#############################################################
# uv layer - copy from official distroless image
# Using SHA256 pinning for reproducible builds
# See: https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
#############################################################

FROM ${UV_IMAGE} AS uv-layer

#############################################################
# Base Ubuntu image with system dependencies
#############################################################

FROM mirror.gcr.io/ubuntu:${UBUNTU_VERSION} AS base

ARG UBUNTU_VERSION
ARG PYTHON_VERSION

# Enforce strict error handling: exit on error, unset vars, pipe failures
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# IMPORTANT: OMPI_VERSION must match the ARG OMPI_VERSION in Dockerfile.tools.
# The openmpi-layer provides /install/openmpi-<tag>-ulfm/ which we copy to /opt/.
# A mismatch causes a silent empty copy; validated after COPY below.

# For system installs of uv, UV_PYTHON_INSTALL_DIR ensures Python is in an accessible
# shared path. When the venv is copied to NFS for multi-host use, the Python interpreter
# must be reachable from all nodes. See scripts/install-uv.sh post-install message.

ENV DEBIAN_FRONTEND=noninteractive
ENV OMPI_VERSION=v5.0.7
ENV OMPI_PREFIX=/opt/openmpi-${OMPI_VERSION}-ulfm
ENV PATH="${OMPI_PREFIX}/bin:/usr/local/bin:$PATH"
ENV LD_LIBRARY_PATH=${OMPI_PREFIX}/lib
ENV CPATH=${OMPI_PREFIX}/include
ENV PKG_CONFIG_PATH=${OMPI_PREFIX}/lib/pkgconfig
ENV UV_PYTHON_INSTALL_DIR="/usr/local/share/uv"

COPY --from=uv-layer /uv /uvx /usr/local/bin/
COPY --from=cmake-layer /install/ /usr/local/
COPY --from=zstd-layer /install/ /usr/local/
COPY --from=openmpi-layer /install/ /opt/

RUN test -x /usr/local/bin/cmake || { echo "ERROR: cmake not found in cmake-layer" >&2; exit 1; } && \
    test -x /usr/local/bin/zstd || { echo "ERROR: zstd not found in zstd-layer" >&2; exit 1; } && \
    test -d "${OMPI_PREFIX}/bin" || \
    { echo "ERROR: ${OMPI_PREFIX}/bin not found. OMPI_VERSION (${OMPI_VERSION}) likely differs from Dockerfile.tools ARG OMPI_VERSION." >&2; exit 1; }

# Auto-detect Python version from Ubuntu version if not explicitly provided
RUN DETECTED_PYTHON_VERSION="${PYTHON_VERSION}"; \
    if [ -z "${DETECTED_PYTHON_VERSION}" ]; then \
    if [ "${UBUNTU_VERSION}" = "24.04" ]; then \
    DETECTED_PYTHON_VERSION="3.12"; \
    else \
    DETECTED_PYTHON_VERSION="3.10"; \
    fi; \
    fi; \
    echo "Installing Python ${DETECTED_PYTHON_VERSION} for Ubuntu ${UBUNTU_VERSION}"; \
    umask 000 && uv python install "${DETECTED_PYTHON_VERSION}"

# Install runtime deps (SFPI already copied from tool image above)
COPY /install_dependencies.sh /opt/tt_metal_infra/scripts/docker/install_dependencies.sh
COPY /tt_metal/sfpi-info.sh /opt/tt_metal_infra/scripts/docker/sfpi-info.sh
COPY /tt_metal/sfpi-version /opt/tt_metal_infra/scripts/docker/sfpi-version
RUN /bin/bash /opt/tt_metal_infra/scripts/docker/install_dependencies.sh --docker

# Copy SFPI from pre-built tool image
RUN --mount=from=sfpi-layer,source=/install,target=/sfpi-staging \
    mkdir -p /opt/tenstorrent && \
    cp -a /sfpi-staging/opt/tenstorrent/sfpi /opt/tenstorrent/ && \
    test -d /opt/tenstorrent/sfpi || { echo "ERROR: sfpi not found in sfpi-layer" >&2; exit 1; }

#############################################################
# CI Build light image (no Python venv layers)
#############################################################

FROM base AS ci-build-light
LABEL org.opencontainers.image.source="https://github.com/tenstorrent/tt-metal"
LABEL org.opencontainers.image.description="TT-Metalium CI build environment without pre-built Python venv layers"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.documentation="https://docs.tenstorrent.com/"

ENV CCACHE_TEMPDIR=/tmp/ccache

# Copy pre-built tools from GHCR tool images
# These images are built by Dockerfile.tools and pushed by build-docker-artifact.yaml
COPY --from=ccache-layer /install/ /usr/local/
COPY --from=doxygen-layer /install/ /usr/local/
COPY --from=clangbuildanalyzer-layer /install/ /usr/local/
COPY --from=yq-layer /install/bin/yq /usr/local/bin/
COPY --from=mold-layer /install/ /usr/local/
RUN test -x /usr/local/bin/ccache || { echo "ERROR: ccache not found in ccache-layer" >&2; exit 1; } && \
    test -x /usr/local/bin/doxygen || { echo "ERROR: doxygen not found in doxygen-layer" >&2; exit 1; } && \
    test -x /usr/local/bin/yq || { echo "ERROR: yq not found in yq-layer" >&2; exit 1; } && \
    test -x /usr/local/bin/mold || { echo "ERROR: mold not found in mold-layer" >&2; exit 1; }

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
    unzip \
    wget \
    zstd \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

#############################################################
# CI Build image
#############################################################

FROM ci-build-light AS ci-build
LABEL org.opencontainers.image.source="https://github.com/tenstorrent/tt-metal"
LABEL org.opencontainers.image.description="TT-Metalium CI build environment for compiling tt-metal"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.documentation="https://docs.tenstorrent.com/"

ENV PYTHON_ENV_DIR=/opt/venv
ENV PATH="$PYTHON_ENV_DIR/bin:$PATH"
ENV VIRTUAL_ENV="$PYTHON_ENV_DIR"

# Copy pre-built ci-build venv from tool image
# Set umask 000 to allow all users to write (required for uv pip install at runtime)
# SECURITY NOTE: This enables any user in the container to modify the Python environment.
RUN --mount=from=ci-build-venv-layer,source=/install,target=/venv-staging \
    umask 000 && \
    cp -a /venv-staging/venv $PYTHON_ENV_DIR && \
    umask 002 && \
    echo "source $PYTHON_ENV_DIR/bin/activate" >> /etc/bash.bashrc

#############################################################
# CI Test light image (no Python venv layers)
#############################################################

FROM ci-build-light AS ci-test-light
LABEL org.opencontainers.image.source="https://github.com/tenstorrent/tt-metal"
LABEL org.opencontainers.image.description="TT-Metalium CI test environment without pre-built Python venv layers"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.documentation="https://docs.tenstorrent.com/"

ARG UBUNTU_VERSION
ARG TT_SMI_VERSION
ARG TT_METAL_INFRA_DIR=/opt/tt_metal_infra

# Install tt-smi from GitHub releases and make it executable for all users
# (notably the default non-root container user). Verify the installed version
# matches TT_SMI_VERSION in the same command.
RUN curl -fsSL -o /usr/local/bin/tt-smi \
    "https://github.com/tenstorrent/tt-smi/releases/download/v${TT_SMI_VERSION}/tt-smi-${TT_SMI_VERSION}-ubuntu-${UBUNTU_VERSION}" && \
    chmod a+rx /usr/local/bin/tt-smi && \
    tt-smi --version | grep -q "${TT_SMI_VERSION}"

# Create directories for infra (needed even with pre-built venv for requirements file reference)
RUN mkdir -p ${TT_METAL_INFRA_DIR}/tt-metal/docs/ && \
    mkdir -p ${TT_METAL_INFRA_DIR}/tt-metal/tests/sweep_framework/ && \
    mkdir -p ${TT_METAL_INFRA_DIR}/tt-metal/tt_metal/python_env/ && \
    mkdir -p ${TT_METAL_INFRA_DIR}/tt-metal/models/tt_dit/utils/ && \
    mkdir -p ${TT_METAL_INFRA_DIR}/tools/triage/

# Copy requirements files (needed for fallback install if no pre-built venv)
COPY /docs/requirements-docs.txt ${TT_METAL_INFRA_DIR}/tt-metal/docs/.
COPY /tests/sweep_framework/requirements-sweeps.txt ${TT_METAL_INFRA_DIR}/tt-metal/tests/sweep_framework/.
COPY /tt_metal/python_env/requirements-dev.txt ${TT_METAL_INFRA_DIR}/tt-metal/tt_metal/python_env/.
COPY /models/tt_dit/utils/vbench-requirements.txt ${TT_METAL_INFRA_DIR}/tt-metal/models/tt_dit/utils/vbench-requirements.txt
COPY /tools/triage/requirements.txt ${TT_METAL_INFRA_DIR}/tools/triage/requirements.txt

#############################################################
# CI Test image
#############################################################

FROM ci-test-light AS ci-test
LABEL org.opencontainers.image.source="https://github.com/tenstorrent/tt-metal"
LABEL org.opencontainers.image.description="TT-Metalium CI test environment with Python dependencies"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.documentation="https://docs.tenstorrent.com/"

ENV PYTHON_ENV_DIR=/opt/venv
ENV PATH="$PYTHON_ENV_DIR/bin:$PATH"
ENV VIRTUAL_ENV="$PYTHON_ENV_DIR"

# Copy pre-built ci-test venv. This layer already includes the ci-build venv packages.
RUN --mount=from=ci-test-venv-layer,source=/install,target=/venv-staging \
    umask 000 && \
    cp -a /venv-staging/venv $PYTHON_ENV_DIR && \
    umask 002 && \
    echo "source $PYTHON_ENV_DIR/bin/activate" >> /etc/bash.bashrc

#############################################################
# Development light image (no Python venv layers)
#############################################################

FROM ci-test-light AS dev-light
LABEL org.opencontainers.image.source="https://github.com/tenstorrent/tt-metal"
LABEL org.opencontainers.image.description="TT-Metalium development environment with full toolchain, without pre-built Python venv layers"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.documentation="https://docs.tenstorrent.com/"

ARG UBUNTU_VERSION

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
    gh \
    less \
    libmpfr-dev \
    nano \
    openssh-server \
    vim \
    libgl1 \
    ipmitool \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Configure SSH for MPI multihost jobs readiness
# To be safe, keep this **after** the apt install to avoid openssh-server mangling configs
RUN adduser --uid 1001 --shell /bin/bash --disabled-password --gecos "" user && \
    usermod -aG sudo user && \
    echo 'user ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/user && chmod 0440 /etc/sudoers.d/user && \
    mkdir -p /run/sshd && \
    echo "StrictModes no" >> /etc/ssh/sshd_config

#############################################################
# Development image
#############################################################

FROM dev-light AS dev
LABEL org.opencontainers.image.source="https://github.com/tenstorrent/tt-metal"
LABEL org.opencontainers.image.description="TT-Metalium development environment with full toolchain"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.documentation="https://docs.tenstorrent.com/"

ENV PYTHON_ENV_DIR=/opt/venv
ENV PATH="$PYTHON_ENV_DIR/bin:$PATH"
ENV VIRTUAL_ENV="$PYTHON_ENV_DIR"

# Copy pre-built ci-test venv. This layer already includes the ci-build venv packages.
RUN --mount=from=ci-test-venv-layer,source=/install,target=/venv-staging \
    umask 000 && \
    cp -a /venv-staging/venv $PYTHON_ENV_DIR && \
    umask 002 && \
    echo "source $PYTHON_ENV_DIR/bin/activate" >> /etc/bash.bashrc

# IWYU (Include What You Use) could be useful to developers.
# Not currently included; install script ready at dockerfile/scripts/install-iwyu.sh
# To add: create iwyu tool target in Dockerfile.tools using install-iwyu.sh

#############################################################
# Release image
#############################################################

FROM base AS release
LABEL org.opencontainers.image.source="https://github.com/tenstorrent/tt-metal"
LABEL org.opencontainers.image.description="TT-Metalium runtime environment for deployment"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.documentation="https://docs.tenstorrent.com/"

ARG PYTHON_VERSION

# Set up virtual environment
ENV PYTHON_ENV_DIR=/opt/venv
ENV PATH="$PYTHON_ENV_DIR/bin:$PATH"
ENV VIRTUAL_ENV="$PYTHON_ENV_DIR"

ARG WHEEL_FILENAME
ADD $WHEEL_FILENAME $WHEEL_FILENAME

# Set umask 000 to allow all users to write (required for uv pip install at runtime)
# SECURITY NOTE: This enables any user in the container to modify the Python environment.
# This is acceptable for single-user containers but should be reconsidered for multi-tenant
# or security-sensitive deployments. Consider using user-specific venvs in those cases.
# Patch relocatable venv activate for POSIX sh (Docker RUN uses /bin/sh)
COPY /scripts/patch_activate_posix.sh /opt/tt_metal_infra/scripts/patch_activate_posix.sh
RUN umask 000 && \
    uv venv --python "$PYTHON_VERSION" --link-mode copy --managed-python --relocatable "$PYTHON_ENV_DIR" && \
    /bin/sh /opt/tt_metal_infra/scripts/patch_activate_posix.sh "$PYTHON_ENV_DIR" && \
    umask 002 && \
    echo "source $PYTHON_ENV_DIR/bin/activate" >> /etc/bash.bashrc && \
    mkdir -p /etc && \
    echo "[global]\nextra-index-url = https://download.pytorch.org/whl/cpu" > /etc/pip.conf && \
    umask 000 && \
    uv pip install $WHEEL_FILENAME

#############################################################
# Release models image
#############################################################

FROM base AS release-models
LABEL org.opencontainers.image.source="https://github.com/tenstorrent/tt-metal"
LABEL org.opencontainers.image.description="TT-Metalium runtime environment with models"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.documentation="https://docs.tenstorrent.com/"

ARG PYTHON_VERSION

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
    && ./create_venv.sh --bundle-python --python-version \"$PYTHON_VERSION\" --env-dir \"$PYTHON_ENV_DIR\" \
    && rm -rf /root/.cache /root/.cargo /tt-metal/.cpmcache" && \
    umask 002 && \
    echo "source $PYTHON_ENV_DIR/bin/activate" >> /etc/bash.bashrc

# Extra system deps
RUN apt-get update && apt-get install -y \
    # required
    gosu \
    libgl1 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
