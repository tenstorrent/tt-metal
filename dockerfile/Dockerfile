
#############################################################

# Accept an argument to specify the Ubuntu version
ARG UBUNTU_VERSION=20.04
FROM public.ecr.aws/ubuntu/ubuntu:${UBUNTU_VERSION} AS base

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime deps
COPY /install_dependencies.sh /opt/tt_metal_infra/scripts/docker/install_dependencies.sh
RUN /bin/bash /opt/tt_metal_infra/scripts/docker/install_dependencies.sh --docker --mode runtime


#############################################################

FROM base AS ci-build

RUN /bin/bash /opt/tt_metal_infra/scripts/docker/install_dependencies.sh --docker --mode build

# Install ccache from upstream; Apt's version for 20.04 predates remote_storage support
# When we drop 20.04, can put this in requirements.txt instead of here.
RUN mkdir -p /usr/local/bin && wget -O /tmp/ccache.tar.xz https://github.com/ccache/ccache/releases/download/v4.10.2/ccache-4.10.2-linux-x86_64.tar.xz && \
    tar -xf /tmp/ccache.tar.xz -C /usr/local/bin --strip-components=1 && \
    rm /tmp/ccache.tar.xz

# Default on ubuntu-22.04 is 1.9.1
ARG DOXYGEN_VERSION=1.9.6
RUN mkdir -p /tmp/doxygen \
    && wget -O /tmp/doxygen/doxygen-${DOXYGEN_VERSION}.linux.bin.tar.gz "https://www.doxygen.nl/files/doxygen-${DOXYGEN_VERSION}.linux.bin.tar.gz" \
    && tar -xzf /tmp/doxygen/doxygen-${DOXYGEN_VERSION}.linux.bin.tar.gz -C /tmp/doxygen --strip-components=1 \
    && make -C /tmp/doxygen -j$(nproc) \
    && make -C /tmp/doxygen install \
    && rm -rf /tmp/doxygen

# Add ClangBuildAnalyzer
RUN mkdir -p /tmp/cba \
    && wget -O /tmp/cba/cba.tar.gz https://github.com/aras-p/ClangBuildAnalyzer/archive/refs/tags/v1.6.0.tar.gz \
    && tar -xzf /tmp/cba/cba.tar.gz -C /tmp/cba --strip-components=1 \
    && cmake -S /tmp/cba/ -B /tmp/cba/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/cba/build \
    && cmake --install /tmp/cba/build \
    && rm -rf /tmp/cba

# Setup Env variables to setup Python Virtualenv - Install TT-Metal Python deps
ENV TT_METAL_INFRA_DIR=/opt/tt_metal_infra
ENV PYTHON_ENV_DIR=${TT_METAL_INFRA_DIR}/tt-metal/python_env

# Create directories for infra
RUN mkdir -p ${TT_METAL_INFRA_DIR}/tt-metal/docs/
RUN mkdir -p ${TT_METAL_INFRA_DIR}/tt-metal/tests/sweep_framework/
RUN mkdir -p ${TT_METAL_INFRA_DIR}/tt-metal/tt_metal/python_env/

# Copy requirements from tt-metal folders with requirements.txt docs
COPY /docs/requirements-docs.txt ${TT_METAL_INFRA_DIR}/tt-metal/docs/.
# Copy requirements from tt-metal folders for sweeps (requirements-sweeps.txt)
COPY /tests/sweep_framework/requirements-sweeps.txt ${TT_METAL_INFRA_DIR}/tt-metal/tests/sweep_framework/.
COPY /tt_metal/python_env/requirements-dev.txt ${TT_METAL_INFRA_DIR}/tt-metal/tt_metal/python_env/.

RUN python3 -m pip config set global.extra-index-url https://download.pytorch.org/whl/cpu \
    && python3 -m pip install setuptools wheel

RUN python3 -m pip install -r ${TT_METAL_INFRA_DIR}/tt-metal/tt_metal/python_env/requirements-dev.txt
RUN python3 -m pip install -r ${TT_METAL_INFRA_DIR}/tt-metal/docs/requirements-docs.txt

# If any file in scripts changes, cache is invalidated
COPY /scripts /opt/tt_metal_infra/scripts

# Install extra ci apt requirements
RUN apt-get -y update \
    && xargs -a /opt/tt_metal_infra/scripts/docker/requirements_ci.txt apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

ENV CCACHE_TEMPDIR=/tmp/ccache

#############################################################

FROM ci-build AS ci-test

# Setup Env variables to setup Python Virtualenv - Install TT-Metal Python deps
ENV TT_METAL_INFRA_DIR=/opt/tt_metal_infra
ENV PYTHON_ENV_DIR=${TT_METAL_INFRA_DIR}/tt-metal/python_env

# Create directories for infra
RUN mkdir -p ${TT_METAL_INFRA_DIR}/tt-metal/docs/
RUN mkdir -p ${TT_METAL_INFRA_DIR}/tt-metal/tests/sweep_framework/
RUN mkdir -p ${TT_METAL_INFRA_DIR}/tt-metal/tt_metal/python_env/
# Copy requirements from tt-metal folders with requirements.txt docs
COPY /docs/requirements-docs.txt ${TT_METAL_INFRA_DIR}/tt-metal/docs/.
# Copy requirements from tt-metal folders for sweeps (requirements-sweeps.txt)
COPY /tests/sweep_framework/requirements-sweeps.txt ${TT_METAL_INFRA_DIR}/tt-metal/tests/sweep_framework/.
COPY /tt_metal/python_env/requirements-dev.txt ${TT_METAL_INFRA_DIR}/tt-metal/tt_metal/python_env/.

RUN python3 -m pip config set global.extra-index-url https://download.pytorch.org/whl/cpu && \
    python3 -m pip install setuptools wheel && \
    python3 -m pip install -r ${TT_METAL_INFRA_DIR}/tt-metal/tt_metal/python_env/requirements-dev.txt && \
    python3 -m pip install -r ${TT_METAL_INFRA_DIR}/tt-metal/docs/requirements-docs.txt

#############################################################

FROM ci-test AS dev

# Need this to build GDB
RUN apt-get -y update \
    && apt-get install -y libmpfr-dev \
    && rm -rf /var/lib/apt/lists/*

# Install the gdb that is compatible with clang-17
RUN apt-get remove -y gdb || true \
    && mkdir -p /tmp/gdb-build && cd /tmp/gdb-build/ \
    && wget -O /tmp/gdb-build/gdb.tar.gz https://ftp.gnu.org/gnu/gdb/gdb-14.2.tar.gz \
    && tar -xvf /tmp/gdb-build/gdb.tar.gz -C /tmp/gdb-build --strip-components=1 \
    && /tmp/gdb-build/configure --prefix=/usr/local \
    && make -j$(nproc) \
    && make install \
    && rm -rf /tmp/gdb-build

# Install dev deps
RUN apt-get -y update \
    && xargs -a /opt/tt_metal_infra/scripts/docker/requirements_dev.txt apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

#############################################################

FROM base AS release

ARG WHEEL_FILENAME
ADD $WHEEL_FILENAME $WHEEL_FILENAME
RUN pip3 install $WHEEL_FILENAME
