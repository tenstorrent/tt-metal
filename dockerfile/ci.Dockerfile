
ARG UBUNTU_VERSION=20.04
ARG TAG=latest
FROM ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-${UBUNTU_VERSION}-amd64-base:${TAG}

# Install ccache from upstream; Apt's version for 20.04 predates remote_storage support
# When we drop 20.04, can put this in requirements.txt instead of here.
RUN wget -O /tmp/ccache.tar.xz https://github.com/ccache/ccache/releases/download/v4.10.2/ccache-4.10.2-linux-x86_64.tar.xz && \
    tar -xf /tmp/ccache.tar.xz -C /usr/local/bin --strip-components=1 && \
    rm /tmp/ccache.tar.xz

# Script is already there from base layer
# COPY /install_dependencies.sh /opt/tt_metal_infra/scripts/docker/install_dependencies.sh
RUN /bin/bash /opt/tt_metal_infra/scripts/docker/install_dependencies.sh --docker --mode build

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

# If any file in scripts changes, cache is invalidated
COPY /scripts /opt/tt_metal_infra/scripts

# Install extra ci apt requirements
RUN apt-get -y update \
    && xargs -a /opt/tt_metal_infra/scripts/docker/requirements-${UBUNTU_VERSION}.txt apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*
