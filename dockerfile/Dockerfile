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
#   docker buildx bake -f dockerfile/docker-bake.hcl ci-test
#   UBUNTU_VERSION=24.04 PYTHON_VERSION=3.12 docker buildx bake -f dockerfile/docker-bake.hcl dev
#   docker buildx bake -f dockerfile/docker-bake.hcl --print dev   # dry run
#
# Or use the wrapper script:
#   ./dockerfile/build-local.sh dev
#   ./dockerfile/build-local.sh --help
#
# Available targets: ci-build, ci-test, dev, release, release-models
# =============================================================================

#############################################################
# Build arguments
#############################################################

ARG UBUNTU_VERSION=22.04
# PYTHON_VERSION can be explicitly set or auto-detected from UBUNTU_VERSION:
# Ubuntu 22.04 -> Python 3.10
# Ubuntu 24.04 -> Python 3.12
ARG PYTHON_VERSION=""
ARG UV_IMAGE=ghcr.io/astral-sh/uv@sha256:9a23023be68b2ed09750ae636228e903a54a05ea56ed03a934d00fe9fbeded4b

#############################################################
# Tool and venv layers (resolved by docker-bake.hcl contexts)
#
# These FROM scratch stubs are overridden by Bake:
#   - Locally: Bake builds Dockerfile.tools/Dockerfile.python targets first
#   - In CI:   Bake substitutes pre-built GHCR images via --set contexts
# Without Bake, these resolve to empty scratch images (same as before).
#############################################################

FROM scratch AS ccache-layer
FROM scratch AS mold-layer
FROM scratch AS doxygen-layer
FROM scratch AS cba-layer
FROM scratch AS gdb-layer
FROM scratch AS cmake-layer
FROM scratch AS yq-layer
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

# IMPORTANT: Keep this in sync with OMPI_TAG in Dockerfile.tools
# OpenMPI configuration - must match Dockerfile.tools OMPI_TAG
# The openmpi-layer provides /install/openmpi-<tag>-ulfm/ which we copy to /opt/

# For system installs of uv, UV_PYTHON_INSTALL_DIR ensures Python is in an accessible
# shared path. When the venv is copied to NFS for multi-host use, the Python interpreter
# must be reachable from all nodes. See scripts/install-uv.sh post-install message.

ENV DEBIAN_FRONTEND=noninteractive \
    OMPI_TAG=v5.0.7 \
    OMPI_PREFIX=/opt/openmpi-${OMPI_TAG}-ulfm

ENV PATH="${OMPI_PREFIX}/bin:/usr/local/bin:$PATH" \
    LD_LIBRARY_PATH=${OMPI_PREFIX}/lib \
    CPATH=${OMPI_PREFIX}/include \
    PKG_CONFIG_PATH=${OMPI_PREFIX}/lib/pkgconfig \
    UV_PYTHON_INSTALL_DIR="/usr/local/share/uv"

COPY --from=uv-layer /uv /uvx /usr/local/bin/
COPY --from=cmake-layer /install/ /usr/local/
COPY --from=openmpi-layer /install/ /opt/

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
LABEL org.opencontainers.image.source="https://github.com/tenstorrent/tt-metal"
LABEL org.opencontainers.image.description="TT-Metalium CI build environment for compiling tt-metal"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.documentation="https://docs.tenstorrent.com/"

# Ensure /usr/local/bin is in PATH so CMake can find mold (ld.mold)
ENV PYTHON_ENV_DIR=/opt/venv

ENV PATH="$PYTHON_ENV_DIR/bin:$PATH" \
    VIRTUAL_ENV="$PYTHON_ENV_DIR" \
    CCACHE_TEMPDIR=/tmp/ccache

# Copy pre-built tools from GHCR tool images
# These images are built by Dockerfile.tools and pushed by build-docker-artifact.yaml
COPY --from=ccache-layer /install/bin/ccache /usr/local/bin/
COPY --from=doxygen-layer /install/ /usr/local/
COPY --from=cba-layer /install/ /usr/local/
COPY --from=yq-layer /install/bin/yq /usr/local/bin/
COPY --from=mold-layer /install/ /usr/local/

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
LABEL org.opencontainers.image.source="https://github.com/tenstorrent/tt-metal"
LABEL org.opencontainers.image.description="TT-Metalium CI test environment with Python dependencies"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.documentation="https://docs.tenstorrent.com/"

ARG UBUNTU_VERSION
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
    cp -ua /venv-staging/venv $PYTHON_ENV_DIR
#rm -rf $PYTHON_ENV_DIR # try just copying in place

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
LABEL org.opencontainers.image.source="https://github.com/tenstorrent/tt-metal"
LABEL org.opencontainers.image.description="TT-Metalium development environment with full toolchain"
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

# Configure SSH for MPI multihost jobs readiness
RUN adduser --uid 1001 --shell /bin/bash user && \
    usermod -aG sudo user && \
    echo 'user ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/user && chmod 0440 /etc/sudoers.d/user && \
    mkdir -p /run/sshd && \
    echo "StrictModes no" | sudo tee -a /etc/ssh/sshd_config

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
LABEL org.opencontainers.image.source="https://github.com/tenstorrent/tt-metal"
LABEL org.opencontainers.image.description="TT-Metalium runtime environment for deployment"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.documentation="https://docs.tenstorrent.com/"

ARG PYTHON_VERSION

# Set up virtual environment
ENV PYTHON_ENV_DIR=/opt/venv

ENV PATH="$PYTHON_ENV_DIR/bin:$PATH" \
    VIRTUAL_ENV="$PYTHON_ENV_DIR"

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
ENV PYTHON_ENV_DIR=/opt/venv \
    TT_METAL_HOME=/tt-metal \
    PYTHONPATH=/tt-metal

ENV PATH="$PYTHON_ENV_DIR/bin:$PATH" \
    VIRTUAL_ENV="$PYTHON_ENV_DIR"

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
    && ./create_venv.sh --bundle-python --python-version "$PYTHON_VERSION" --env-dir "$PYTHON_ENV_DIR" \
    && source "$PYTHON_ENV_DIR/bin/activate" \
    && rm -rf /root/.cache /root/.cargo /tt-metal/.cpmcache" && \
    umask 002 && \
    echo "source $PYTHON_ENV_DIR/bin/activate" >> /etc/bash.bashrc

# Ensure the virtual environment is used for all Python-related commands
ENV PATH="$PYTHON_ENV_DIR/bin:$PATH" \
    VIRTUAL_ENV="$PYTHON_ENV_DIR"

# Extra system deps
RUN apt-get update && apt-get install -y \
    # required
    gosu \
    libgl1 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
