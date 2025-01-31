
ARG UBUNTU_VERSION=20.04
ARG TAG=latest
FROM ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-${UBUNTU_VERSION}-amd64-base:${TAG}

ARG UBUNTU_VERSION

# Script is already there from base layer
# COPY /install_dependencies.sh /opt/tt_metal_infra/scripts/docker/install_dependencies.sh
RUN /bin/bash /opt/tt_metal_infra/scripts/docker/install_dependencies.sh --docker --mode build

# Install ccache from upstream; Apt's version for 20.04 predates remote_storage support
# When we drop 20.04, can put this in requirements.txt instead of here.
RUN mkdir -p /usr/local/bin && wget -O /tmp/ccache.tar.xz https://github.com/ccache/ccache/releases/download/v4.10.2/ccache-4.10.2-linux-x86_64.tar.xz && \
    tar -xf /tmp/ccache.tar.xz -C /usr/local/bin --strip-components=1 && \
    rm /tmp/ccache.tar.xz

# I don't know why we can't get this from apt
# Lets remove this later if we can
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
    && cmake -S /tmp/cba/ -B /tmp/cba/build -DBUILD_TYPE=Release \
    && cmake --build /tmp/cba/build \
    && cmake --install /tmp/cba/build \
    && rm -rf /tmp/cba

# If any file in scripts changes, cache is invalidated
COPY /scripts /opt/tt_metal_infra/scripts

# Install extra ci apt requirements
RUN apt-get -y update \
    && xargs -a /opt/tt_metal_infra/scripts/docker/requirements-${UBUNTU_VERSION}.txt apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*
