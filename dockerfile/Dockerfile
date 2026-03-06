# TT-Metalium Dockerfile

#############################################################

# Global build arguments (defined before FROM to be available in all stages)
ARG UBUNTU_VERSION=22.04
# PYTHON_VERSION can be explicitly set or auto-detected from UBUNTU_VERSION:
# Ubuntu 22.04 -> Python 3.10
# Ubuntu 24.04 -> Python 3.12
ARG PYTHON_VERSION=""

FROM ghcr.io/astral-sh/uv@sha256:9a23023be68b2ed09750ae636228e903a54a05ea56ed03a934d00fe9fbeded4b AS uv-layer

FROM mirror.gcr.io/ubuntu:${UBUNTU_VERSION} AS base

ARG UBUNTU_VERSION
ARG PYTHON_VERSION

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/bin:$PATH"

COPY --from=uv-layer /uv /uvx /usr/local/bin/

# For system installs of uv, UV_PYTHON_INSTALL_DIR ensures Python is in an accessible
# shared path. When the venv is copied to NFS for multi-host use, the Python interpreter
# must be reachable from all nodes. See scripts/install-uv.sh post-install message.
ENV UV_PYTHON_INSTALL_DIR="/usr/local/share/uv"

# Auto-detect Python version from Ubuntu version if not explicitly provided
RUN set -e; \
    DETECTED_PYTHON_VERSION="${PYTHON_VERSION}"; \
    if [ -z "${DETECTED_PYTHON_VERSION}" ]; then \
    if [ "${UBUNTU_VERSION}" = "24.04" ]; then \
    DETECTED_PYTHON_VERSION="3.12"; \
    else \
    DETECTED_PYTHON_VERSION="3.10"; \
    fi; \
    fi; \
    echo "Installing Python ${DETECTED_PYTHON_VERSION} for Ubuntu ${UBUNTU_VERSION}"; \
    umask 000 && uv python install "${DETECTED_PYTHON_VERSION}"

# Install runtime deps
COPY /install_dependencies.sh /opt/tt_metal_infra/scripts/docker/install_dependencies.sh
COPY /tt_metal/sfpi-info.sh /opt/tt_metal_infra/scripts/docker/sfpi-info.sh
COPY /tt_metal/sfpi-version /opt/tt_metal_infra/scripts/docker/sfpi-version
RUN /bin/bash /opt/tt_metal_infra/scripts/docker/install_dependencies.sh --docker


#############################################################

FROM base AS ci-build

ARG PYTHON_VERSION

# Layer ordering: stable binary downloads first, then source builds, then apt
# packages (which change with Ubuntu version / LLVM releases), then pip packages.

# 1. Pin-versioned binary downloads (very stable, rarely invalidated)
RUN mkdir -p /usr/local/bin && wget -O /tmp/ccache.tar.xz https://github.com/ccache/ccache/releases/download/v4.10.2/ccache-4.10.2-linux-x86_64.tar.xz && \
    tar -xf /tmp/ccache.tar.xz -C /usr/local/bin --strip-components=1 && \
    rm /tmp/ccache.tar.xz

ARG MOLD_VERSION=2.40.4
ARG MOLD_SHA256=4c999e19ffa31afa5aa429c679b665d5e2ca5a6b6832ad4b79668e8dcf3d8ec1
RUN wget -O /tmp/mold.tar.gz https://github.com/rui314/mold/releases/download/v${MOLD_VERSION}/mold-${MOLD_VERSION}-x86_64-linux.tar.gz && \
    echo "${MOLD_SHA256}  /tmp/mold.tar.gz" | sha256sum -c - && \
    tar -xzf /tmp/mold.tar.gz -C /usr/local --strip-components=1 && \
    rm /tmp/mold.tar.gz && \
    mold --version

ENV PATH="/usr/local/bin:$PATH"

# 2. Source builds from pinned versions (stable)
ARG DOXYGEN_VERSION=1.9.6
RUN mkdir -p /tmp/doxygen \
    && wget -O /tmp/doxygen/doxygen-${DOXYGEN_VERSION}.linux.bin.tar.gz "https://www.doxygen.nl/files/doxygen-${DOXYGEN_VERSION}.linux.bin.tar.gz" \
    && tar -xzf /tmp/doxygen/doxygen-${DOXYGEN_VERSION}.linux.bin.tar.gz -C /tmp/doxygen --strip-components=1 \
    && make -C /tmp/doxygen -j$(nproc) \
    && make -C /tmp/doxygen install \
    && rm -rf /tmp/doxygen

RUN mkdir -p /tmp/cba \
    && wget -O /tmp/cba/cba.tar.gz https://github.com/aras-p/ClangBuildAnalyzer/archive/refs/tags/v1.6.0.tar.gz \
    && tar -xzf /tmp/cba/cba.tar.gz -C /tmp/cba --strip-components=1 \
    && cmake -S /tmp/cba/ -B /tmp/cba/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/cba/build \
    && cmake --install /tmp/cba/build \
    && rm -rf /tmp/cba

# 3. Apt packages (changes when LLVM/Ubuntu versions bump)
# Note: llvm-20 is required for llvm-ar-20, which CMake needs for LTO support detection
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
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
    libgtest-dev

# Install yq (YAML processor) for CI workflows
# Note: mikefarah/yq is not available in apt, download binary from GitHub releases
ARG YQ_VERSION=v4.44.6
RUN wget -q "https://github.com/mikefarah/yq/releases/download/${YQ_VERSION}/yq_linux_amd64" -O /usr/local/bin/yq && \
    chmod +x /usr/local/bin/yq

# 4. Python venv and pip packages (changes with dependency updates)
ENV PYTHON_ENV_DIR=/opt/venv
# SECURITY NOTE: umask 000 allows any container user to modify the Python environment.
# Acceptable for single-user containers; reconsider for multi-tenant deployments.
RUN umask 000 && \
    uv venv --python "$PYTHON_VERSION" --link-mode copy --managed-python --relocatable "$PYTHON_ENV_DIR"

# Patch relocatable venv activate for POSIX sh (Docker RUN uses /bin/sh; activate uses $BASH_SOURCE otherwise)
COPY /scripts/patch_activate_posix.sh /opt/tt_metal_infra/scripts/patch_activate_posix.sh
RUN /bin/sh /opt/tt_metal_infra/scripts/patch_activate_posix.sh "$PYTHON_ENV_DIR"

ENV PATH="$PYTHON_ENV_DIR/bin:$PATH"
ENV VIRTUAL_ENV="$PYTHON_ENV_DIR"

RUN echo "source $PYTHON_ENV_DIR/bin/activate" >> /etc/bash.bashrc

RUN --mount=type=cache,target=/root/.cache/uv \
    umask 000 && uv pip install --no-cache-dir uv build numpy setuptools==80

RUN git config --global --add safe.directory '*'

ENV CCACHE_TEMPDIR=/tmp/ccache

#############################################################

FROM ci-build AS ci-test

ARG UBUNTU_VERSION
ARG TT_METAL_INFRA_DIR=/opt/tt_metal_infra

# Create directories for infra and copy all requirement files
RUN mkdir -p ${TT_METAL_INFRA_DIR}/tt-metal/docs/ \
    ${TT_METAL_INFRA_DIR}/tt-metal/tests/sweep_framework/ \
    ${TT_METAL_INFRA_DIR}/tt-metal/tt_metal/python_env/ \
    ${TT_METAL_INFRA_DIR}/tools/triage/

COPY /docs/requirements-docs.txt ${TT_METAL_INFRA_DIR}/tt-metal/docs/.
COPY /tests/sweep_framework/requirements-sweeps.txt ${TT_METAL_INFRA_DIR}/tt-metal/tests/sweep_framework/.
COPY /tt_metal/python_env/requirements-dev.txt ${TT_METAL_INFRA_DIR}/tt-metal/tt_metal/python_env/.
COPY /tt_metal/python_env/requirements-core.txt ${TT_METAL_INFRA_DIR}/tt-metal/tt_metal/python_env/.
COPY /tt_metal/python_env/requirements-dev-tools.txt ${TT_METAL_INFRA_DIR}/tt-metal/tt_metal/python_env/.
COPY /tt_metal/python_env/requirements-models.txt ${TT_METAL_INFRA_DIR}/tt-metal/tt_metal/python_env/.
COPY /tt_metal/python_env/requirements-notebooks.txt ${TT_METAL_INFRA_DIR}/tt-metal/tt_metal/python_env/.
COPY /tools/triage/requirements.txt ${TT_METAL_INFRA_DIR}/tools/triage/requirements.txt

# Install all Python packages in one pass (requirements-dev.txt composes all tiers)
RUN --mount=type=cache,target=/root/.cache/uv \
    umask 000 && \
    uv pip install --no-cache --index-url https://download.pytorch.org/whl/cpu torch && \
    uv pip install --index-strategy unsafe-best-match --no-cache --no-build-isolation -r ${TT_METAL_INFRA_DIR}/tt-metal/tt_metal/python_env/requirements-dev.txt && \
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

FROM ci-test AS dev

ARG UBUNTU_VERSION
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    acl \
    emacs \
    gdb \
    less \
    libmpfr-dev \
    nano \
    openssh-server \
    vim \
    libgl1 \
    ipmitool

# Remove gdb if we aren't on Ubuntu 24.04
# 24.04 has gdb 15.1 by default, lets give that a chance before we decide we need to build something
RUN set -eux; echo "Ubuntu version: $UBUNTU_VERSION"
RUN if [ "$UBUNTU_VERSION" != "24.04" ]; then \
    echo "Removing system gdb and building custom GDB for Ubuntu $UBUNTU_VERSION" && \
    apt-get remove -y gdb && \
    mkdir -p /tmp/gdb-build && cd /tmp/gdb-build/ && \
    wget -O /tmp/gdb-build/gdb.tar.xz https://mirror.csclub.uwaterloo.ca/gnu/gdb/gdb-14.2.tar.xz && \
    tar -xvf /tmp/gdb-build/gdb.tar.xz -C /tmp/gdb-build --strip-components=1 && \
    /tmp/gdb-build/configure --prefix=/usr/local && \
    make -j$(nproc) && \
    make install && \
    rm -rf /tmp/gdb-build; \
    else \
    echo "Using system GDB (15.1) for Ubuntu 24.04 - no custom build needed"; \
    fi

COPY /dockerfile/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Configure SSH for MPI multihost jobs readiness
RUN adduser --uid 1001 --shell /bin/bash user
RUN usermod -aG sudo user
RUN echo 'user ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/user && chmod 0440 /etc/sudoers.d/user

RUN mkdir -p /run/sshd
RUN echo "StrictModes no" | sudo tee -a /etc/ssh/sshd_config

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

#############################################################

FROM base AS runtime

ARG PYTHON_VERSION

# Minimal runtime image for upstream consumers and test execution.
# Contains only system deps, Python, uv, and a venv -- no build tools.

ENV PYTHON_ENV_DIR=/opt/venv

RUN umask 000 && \
    uv venv --python "$PYTHON_VERSION" --link-mode copy --managed-python --relocatable "$PYTHON_ENV_DIR"

COPY /scripts/patch_activate_posix.sh /opt/tt_metal_infra/scripts/patch_activate_posix.sh
RUN /bin/sh /opt/tt_metal_infra/scripts/patch_activate_posix.sh "$PYTHON_ENV_DIR"

ENV PATH="$PYTHON_ENV_DIR/bin:$PATH"
ENV VIRTUAL_ENV="$PYTHON_ENV_DIR"

RUN echo "source $PYTHON_ENV_DIR/bin/activate" >> /etc/bash.bashrc

RUN mkdir -p /etc && \
    echo "[global]\nextra-index-url = https://download.pytorch.org/whl/cpu" > /etc/pip.conf

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    openssh-server \
    sudo

#############################################################

FROM base AS release

ARG PYTHON_VERSION

# Set up virtual environment
ENV PYTHON_ENV_DIR=/opt/venv
# SECURITY NOTE: umask 000 allows any container user to modify the Python environment.
# Acceptable for single-user containers; reconsider for multi-tenant deployments.
RUN umask 000 && uv venv --python "$PYTHON_VERSION" --link-mode copy --managed-python --relocatable "$PYTHON_ENV_DIR"

# Patch relocatable venv activate for POSIX sh (Docker RUN uses /bin/sh)
COPY /scripts/patch_activate_posix.sh /opt/tt_metal_infra/scripts/patch_activate_posix.sh
RUN /bin/sh /opt/tt_metal_infra/scripts/patch_activate_posix.sh "$PYTHON_ENV_DIR"

ENV PATH="$PYTHON_ENV_DIR/bin:$PATH"
ENV VIRTUAL_ENV="$PYTHON_ENV_DIR"

RUN echo "source $PYTHON_ENV_DIR/bin/activate" >> /etc/bash.bashrc

RUN mkdir -p /etc && \
    echo "[global]\nextra-index-url = https://download.pytorch.org/whl/cpu" > /etc/pip.conf

ARG WHEEL_FILENAME
ADD $WHEEL_FILENAME $WHEEL_FILENAME
RUN umask 000 && uv pip install $WHEEL_FILENAME

#############################################################

FROM base AS release-models

ARG PYTHON_VERSION

ENV PYTHON_ENV_DIR=/opt/venv
ENV TT_METAL_HOME=/tt-metal
ENV PYTHONPATH=/tt-metal

ARG GIT_REF=main

RUN /bin/bash -c "git clone --filter=blob:none --recurse-submodules --tags \
    https://github.com/tenstorrent/tt-metal.git ${TT_METAL_HOME} \
    && cd ${TT_METAL_HOME} \
    && git checkout ${GIT_REF} \
    && git submodule update --init --recursive"

WORKDIR /tt-metal

RUN umask 000 && /bin/bash -c "./build_metal.sh \
    && ./create_venv.sh --python-version "$PYTHON_VERSION" --env-dir "$PYTHON_ENV_DIR" \
    && source "$PYTHON_ENV_DIR/bin/activate" \
    && rm -rf /root/.cache /root/.cargo /tt-metal/.cpmcache"

ENV PATH="$PYTHON_ENV_DIR/bin:$PATH"
ENV VIRTUAL_ENV="$PYTHON_ENV_DIR"

RUN echo "source $PYTHON_ENV_DIR/bin/activate" >> /etc/bash.bashrc

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y \
    gosu \
    libgl1 \
    libsndfile1
