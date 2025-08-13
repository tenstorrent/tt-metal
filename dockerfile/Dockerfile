# TT-Metalium Dockerfile

#############################################################

# Accept an argument to specify the Ubuntu version
ARG UBUNTU_VERSION=22.04
FROM public.ecr.aws/ubuntu/ubuntu:${UBUNTU_VERSION} AS base

ENV DEBIAN_FRONTEND=noninteractive

ENV RUSTUP_HOME=/usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo
ENV PATH="$CARGO_HOME/bin:$PATH"

# Build a layer with the heavy dependencies that rarely change for de-dupe efficiency

# 1) Install heavy system dependencies first so this layer is cacheable and not invalidated by frequent SFPI changes
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    gpg \
    lsb-release \
    software-properties-common \
    gnupg \
    wget \
    curl \
    jq \
    xz-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Add LLVM and Kitware repositories (for clang-17 and modern CMake)
RUN set -eux; \
    OS_CODENAME="$(. /etc/os-release; echo $UBUNTU_CODENAME)"; \
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -; \
    echo "deb http://apt.llvm.org/${OS_CODENAME}/ llvm-toolchain-${OS_CODENAME}-17 main" > /etc/apt/sources.list.d/llvm-17.list; \
    echo "deb http://apt.llvm.org/${OS_CODENAME}/ llvm-toolchain-${OS_CODENAME}-20 main" > /etc/apt/sources.list.d/llvm-20.list; \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - > /usr/share/keyrings/kitware-archive-keyring.gpg; \
    echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ ${OS_CODENAME} main" > /etc/apt/sources.list.d/kitware.list; \
    apt-get update

# Core build/runtime deps for TT-Metal development (without g++ which varies by Ubuntu version)
RUN apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    pandoc \
    python3-dev \
    python3-pip \
    python3-venv \
    libhwloc-dev \
    libnuma-dev \
    libatomic1 \
    libstdc++6 \
    libboost-dev \
    libtbb-dev \
    libcapstone-dev \
    libc++-17-dev \
    libc++abi-17-dev \
    openmpi-bin \
    libopenmpi-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install correct g++ version depending on Ubuntu version (22.04 -> g++-12, 24.04 -> g++-14)
RUN set -eux; \
    ver=$( . /etc/os-release; echo "$VERSION_ID" ); \
    apt-get update; \
    if [ "$ver" = "24.04" ]; then \
      add-apt-repository -y ppa:ubuntu-toolchain-r/test; \
      apt-get update; \
      apt-get install -y --no-install-recommends g++-14; \
    else \
      apt-get install -y --no-install-recommends g++-12; \
    fi; \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Ensure clang-17 is present so the script will skip re-installing LLVM
RUN apt-get update && apt-get install -y --no-install-recommends \
    clang-17 \
    lld-17 \
    lldb-17 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Pre-install MPI ULFM package (heavy download) so it is cached
RUN set -eux; \
    TMP_DIR="$(mktemp -d)"; \
    DEB_URL="https://github.com/tenstorrent/ompi/releases/download/v5.0.7/openmpi-ulfm_5.0.7-1_amd64.deb"; \
    wget -q -O "$TMP_DIR/openmpi-ulfm.deb" "$DEB_URL"; \
    apt-get update -qq; \
    apt-get install -f -y "$TMP_DIR/openmpi-ulfm.deb"; \
    rm -rf "$TMP_DIR" /var/lib/apt/lists/*

# Pre-install Rust toolchain to cache heavy downloads
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain 1.89.0 -y

# Install SFPI in a separate, frequently-changing layer within base (keeps heavy deps cached; small delta on pulls)
COPY /install_dependencies.sh /opt/tt_metal_infra/scripts/docker/install_dependencies.sh
COPY /tt_metal/sfpi-version.sh /opt/tt_metal_infra/scripts/docker/sfpi-version.sh
RUN /bin/bash /opt/tt_metal_infra/scripts/docker/install_dependencies.sh --docker --sfpi

#############################################################

FROM base AS ci-build

# Install ccache from upstream; Apt's version for 20.04 predates remote_storage support
RUN mkdir -p /usr/local/bin && wget -O /tmp/ccache.tar.xz https://github.com/ccache/ccache/releases/download/v4.10.2/ccache-4.10.2-linux-x86_64.tar.xz && \
    tar -xf /tmp/ccache.tar.xz -C /usr/local/bin --strip-components=1 && \
    rm /tmp/ccache.tar.xz

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

# Install extra ci apt requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    bc \
    libclang-17-dev \
    clang-tidy-17 \
    clang-tidy-20 \
    curl \
    dialog \
    file \
    graphviz \
    jq \
    sudo \
    uuid-runtime \
    wget \
    libgtest-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up virtual environment
ENV PYTHON_ENV_DIR=/opt/venv
RUN python3 -m venv $PYTHON_ENV_DIR

# Ensure the virtual environment is used for all Python-related commands
ENV PATH="$PYTHON_ENV_DIR/bin:$PATH"
ENV VIRTUAL_ENV="$PYTHON_ENV_DIR"

# Ensure the virtual environment is activated on shell startup
RUN echo "source $PYTHON_ENV_DIR/bin/activate" >> /etc/bash.bashrc

# Numpy needed for tt-train to build
RUN python3 -m pip install --no-cache-dir build numpy setuptools

RUN git config --global --add safe.directory '*'

ENV CCACHE_TEMPDIR=/tmp/ccache

#############################################################

FROM ci-build AS ci-test

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

# TODO: Why do we need the chmod?
RUN python3 -m pip config set global.extra-index-url https://download.pytorch.org/whl/cpu && \
    python3 -m pip install --no-cache-dir -r ${TT_METAL_INFRA_DIR}/tt-metal/tt_metal/python_env/requirements-dev.txt && \
    python3 -m pip install --no-cache-dir -r ${TT_METAL_INFRA_DIR}/tt-metal/docs/requirements-docs.txt && \
    chmod -R 777 $VIRTUAL_ENV

#############################################################

FROM ci-test AS dev

# Install dev deps
# libgl1-mesa-glx is needed for Yolo models https://github.com/tenstorrent/tt-metal/pull/19899
# libgl1-mesa-glx does not exist in ubuntu 24.04 which is right around the corner
# switching to libgl1
RUN apt-get update && apt-get install -y --no-install-recommends \
    acl \
    emacs \
    gdb \
    less \
    libmpfr-dev \
    nano \
    openssh-server \
    vim \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# IWYU could be useful to developers
RUN mkdir -p /tmp/iwyu \
    && wget -O /tmp/iwyu/iwyu.tar.gz https://github.com/include-what-you-use/include-what-you-use/archive/refs/tags/0.21.tar.gz \
    && tar -xzf /tmp/iwyu/iwyu.tar.gz -C /tmp/iwyu --strip-components=1 \
    && cmake -S /tmp/iwyu/ -B /tmp/iwyu/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-17 -DCMAKE_CXX_COMPILER=clang++-17 \
    && cmake --build /tmp/iwyu/build --parallel \
    && cmake --install /tmp/iwyu/build \
    && rm -rf /tmp/iwyu

# Remove gdb if we aren't on Ubuntu 24.04
# 24.04 has gdb 15.1 by default, lets give that a chance before we decide we need to build something
RUN [ "$UBUNTU_VERSION" != "24.04" ] && apt-get remove -y gdb || true
RUN [ "$UBUNTU_VERSION" != "24.04" ] \
    && mkdir -p /tmp/gdb-build && cd /tmp/gdb-build/ \
    && wget -O /tmp/gdb-build/gdb.tar.xz https://mirror.csclub.uwaterloo.ca/gnu/gdb/gdb-14.2.tar.xz \
    && tar -xvf /tmp/gdb-build/gdb.tar.xz -C /tmp/gdb-build --strip-components=1 \
    && /tmp/gdb-build/configure --prefix=/usr/local \
    && make -j$(nproc) \
    && make install \
    && rm -rf /tmp/gdb-build

COPY /dockerfile/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

#############################################################

FROM base AS release

# Set up virtual environment
ENV PYTHON_ENV_DIR=/opt/venv
RUN python3 -m venv $PYTHON_ENV_DIR

# Ensure the virtual environment is used for all Python-related commands
ENV PATH="$PYTHON_ENV_DIR/bin:$PATH"
ENV VIRTUAL_ENV="$PYTHON_ENV_DIR"

# Ensure the virtual environment is activated on shell startup
RUN echo "source $PYTHON_ENV_DIR/bin/activate" >> /etc/bash.bashrc

RUN mkdir -p /etc && \
    echo "[global]\nextra-index-url = https://download.pytorch.org/whl/cpu" > /etc/pip.conf

ARG WHEEL_FILENAME
ADD $WHEEL_FILENAME $WHEEL_FILENAME
RUN pip3 install $WHEEL_FILENAME
