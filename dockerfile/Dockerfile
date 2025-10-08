# TT-Metalium Dockerfile

#############################################################

# Accept an argument to specify the Ubuntu version
ARG UBUNTU_VERSION=22.04
FROM mirror.gcr.io/ubuntu:${UBUNTU_VERSION} AS base

ENV DEBIAN_FRONTEND=noninteractive

ENV RUSTUP_HOME=/usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo
ENV PATH="$CARGO_HOME/bin:$PATH"

# Install runtime deps
COPY /install_dependencies.sh /opt/tt_metal_infra/scripts/docker/install_dependencies.sh
COPY /tt_metal/sfpi-info.sh /opt/tt_metal_infra/scripts/docker/sfpi-info.sh
COPY /tt_metal/sfpi-version /opt/tt_metal_infra/scripts/docker/sfpi-version
RUN /bin/bash /opt/tt_metal_infra/scripts/docker/install_dependencies.sh --docker

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
    zstd \
    libgtest-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PYTHON_ENV_DIR=/opt/venv
RUN umask 000 && python3 -m venv $PYTHON_ENV_DIR

# Ensure the virtual environment is used for all Python-related commands
ENV PATH="$PYTHON_ENV_DIR/bin:$PATH"
ENV VIRTUAL_ENV="$PYTHON_ENV_DIR"

# Ensure the virtual environment is activated on shell startup
RUN echo "source $PYTHON_ENV_DIR/bin/activate" >> /etc/bash.bashrc

# Numpy needed for tt-train to build
RUN python3 -m pip install --no-cache-dir build numpy setuptools

ENV CCACHE_TEMPDIR=/tmp/ccache

#############################################################
FROM ci-build as build-artifacts

# Copy files required for build only (omit scripts, docs, tests not needed for build
# misc
#COPY tools /tt-metal/tools
#COPY tt_telemetry /tt-metal/tt_telemetry
#COPY cmake /tt-metal/cmake
#COPY tt-train /tt-metal/tt-train
#COPY tt_stl /tt-metal/tt_stl
#COPY build_metal.sh /tt-metal/build_metal.sh
#COPY CMakeLists.txt /tt-metal/CMakeLists.txt
#COPY CMakePresets.json /tt-metal/CMakePresets.json

# tt-metal
#COPY tt_metal /tt-metal/tt_metal

# ttnn
#COPY ttnn /tt-metal/ttnn

# Tests
#COPY tests/CMakeLists.txt /tt-metal/tests/CMakeLists.txt
#COPY tests/tt_metal /tt-metal/tests/tt_metal
#COPY tests/tt_eager /tt-metal/tests/tt_eager
#COPY tests/ttnn /tt-metal/tests/ttnn

COPY . /tt-metal

WORKDIR /tt-metal

# Override version to not rely at all on git
RUN sed -i 's/set(fallbackVersion "\$Format:%(describe)\$")/set(fallbackVersion "0.0.0")/; s/set(fallbackHash "\$Format:%h\$")/set(fallbackHash "000")/' cmake/version.cmake


ARG SKIP_TT_TRAIN="true"
ARG BUILD_TYPE="Release"
ARG TOOLCHAIN="cmake/x86_64-linux-clang-17-libstdcpp-toolchain.cmake"
ARG TRACY="false"
ARG DISTRIBUTED="true"
ARG PROFILE="false"
ARG BUILD_UMD_TESTS="false"
ARG PUBLISH_ARTIFACT="true"

RUN set -eu && \
    BUILD_ARGS=$([ "${SKIP_TT_TRAIN}" = "true" ] && echo "--build-metal-tests --build-ttnn-tests --build-programming-examples" || echo "--build-all") && \
    BUILD_COMMAND="./build_metal.sh --build-dir build --build-type ${BUILD_TYPE} --toolchain-path ${TOOLCHAIN} ${BUILD_ARGS} --enable-ccache --configure-only" && \
    \
    # Conditionally add Tracy profiler
    if [ "${TRACY}" = "true" ]; then \
        BUILD_COMMAND="${BUILD_COMMAND} --enable-profiler"; \
    fi && \
    \
    # Conditionally disable distributed
    if [ "${DISTRIBUTED}" = "false" ]; then \
        BUILD_COMMAND="${BUILD_COMMAND} --without-distributed"; \
    fi && \
    \
    # Conditionally add profiling options
    if [ "${PROFILE}" = "true" ]; then \
        BUILD_COMMAND="${BUILD_COMMAND} --enable-time-trace --disable-unity-builds"; \
    fi && \
    \
    # Conditionally build UMD tests
    if [ "${BUILD_UMD_TESTS}" = "true" ]; then \
        BUILD_COMMAND="${BUILD_COMMAND} --build-umd-tests"; \
    fi && \
    \
    # Execute the build command
    echo "Build command: ${BUILD_COMMAND}" && \
    ${BUILD_COMMAND}

# this pollutes the system - likely needs --components... (if some missing)
RUN cmake --build build --target install 

# Conditional profiling step
RUN set -eu && \
    if [ "${PROFILE}" = "true" ]; then \
        echo "maxNameLength = 300" > ClangBuildAnalyzer.ini && \
        ClangBuildAnalyzer --all build capture.bin && \
        ClangBuildAnalyzer --analyze capture.bin; \
    fi

RUN cmake --build build --target package
RUN mkdir /build
RUN mv build/*.deb build/*.ddeb /build/

RUN ls -1sh /build/*.deb /build/*.ddeb || true

# Prepare and tar artifacts
RUN set -eu && \
    if [ "${PUBLISH_ARTIFACT}" = "true" ]; then \
        ARTIFACT_PATHS="ttnn/ttnn/*.so build/lib ttnn/ttnn/*.so build/programming_examples build/test build/tools runtime"; \
        \
        if [ "${SKIP_TT_TRAIN}" != "true" ]; then \
            ARTIFACT_PATHS="${ARTIFACT_PATHS} build/tt-train data"; \
        fi && \
        \
        mkdir -p /artifacts && \
        tar -cvhf /artifacts/ttm_any.tar ${ARTIFACT_PATHS}; \
    fi

RUN python3 -m build --wheel -o /artifacts .

FROM scratch AS artifacts
COPY --from=build-artifacts /artifacts /artifacts
COPY --from=build-artifacts /build /build

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

RUN umask 000 && python3 -m pip config set global.extra-index-url https://download.pytorch.org/whl/cpu && \
    python3 -m pip install --no-cache-dir -r ${TT_METAL_INFRA_DIR}/tt-metal/tt_metal/python_env/requirements-dev.txt && \
    python3 -m pip install --no-cache-dir -r ${TT_METAL_INFRA_DIR}/tt-metal/docs/requirements-docs.txt && \
    python3 -m pip install --no-cache-dir -r ${TT_METAL_INFRA_DIR}/tools/triage/requirements.txt

COPY /scripts/install_debugger.sh ${TT_METAL_INFRA_DIR}/scripts/install_debugger.sh
COPY /scripts/ttexalens_ref.txt ${TT_METAL_INFRA_DIR}/scripts/ttexalens_ref.txt
RUN if [ "$UBUNTU_VERSION" = "22.04" ]; then \
    echo "Installing debugger for Ubuntu 22.04..." && \
    ${TT_METAL_INFRA_DIR}/scripts/install_debugger.sh && \
    echo "Debugger installation completed" && \
    python3 -c "import ttexalens; print('ttexalens successfully imported')" || echo "ttexalens import failed"; \
    fi

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
    gdb \
    less \
    libmpfr-dev \
    nano \
    openssh-server \
    vim \
    libgl1 \
    ipmitool \
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

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

## Evaluation image - everything installed & ready to go
FROM dev as evaluation
CMD ["tail", "-f", "/dev/null"]
COPY --from=artifacts /artifacts /artifacts
COPY --from=artifacts /build /build

RUN apt update && apt install -y nlohmann-json3-dev libxtensor-dev

RUN dpkg -i /build/tt-metalium_0.0.0~ubuntu24.04_amd64.deb
RUN dpkg -i /build/tt-nn_0.0.0~ubuntu24.04_amd64.deb
RUN dpkg -i /build/*

ENV LD_LIBRARY_PATH="/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH"
RUN pip3 install /artifacts/ttnn-0.0.0.dev0-cp312-cp312-linux_x86_64.whl

COPY . ./tt-metal

RUN tar -xf /artifacts/ttm_any.tar -C /tt-metal
ENV TT_METAL_HOME=/tt-metal
#ENV TT_METAL_RUNTIME_ROOT=/tt-metal

# Required for MPI
RUN mkdir -p /run/sshd
RUN echo "StrictModes no" >> /etc/ssh/sshd_config

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

#############################################################

FROM base AS release-models

# Set up virtual environment and TT_METAL_HOME
ENV PYTHON_ENV_DIR=/opt/venv
ENV TT_METAL_HOME=/tt-metal
ENV PYTHONPATH=/tt-metal

# Clone tt-metal
RUN /bin/bash -c "git clone --filter=blob:none --recurse-submodules --tags \
    https://github.com/tenstorrent/tt-metal.git ${TT_METAL_HOME} \
    && cd ${TT_METAL_HOME}"

WORKDIR /tt-metal

# Build tt-metal and venv, then clear cache
RUN /bin/bash -c "./build_metal.sh \
    && ./create_venv.sh \
    && source $PYTHON_ENV_DIR/bin/activate \
    && pip install -r models/tt_transformers/requirements.txt \
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
