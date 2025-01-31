
ARG UBUNTU_VERSION=20.04
ARG TAG=latest
FROM ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-${UBUNTU_VERSION}-amd64-ci:${TAG}

# Install the gdb that is compatible with clang-17
RUN apt-get remove -y gdb || true \
    && mkdir -p /tmp/gdb-build \
    && wget -O /tmp/gdb-build/gdb.tar.gz https://ftp.gnu.org/gnu/gdb/gdb-14.2.tar.gz \
    && tar -xvf /tmp/gdb-build/gdb.tar.gz -C /tmp/gdb-build --strip-components=1 \
    && /tmp/gdb-build/configure --prefix=/usr/local \
    && make -C /tmp/gdb-build -j$(nproc) \
    && make -C /tmp/gdb-build install \
    && rm -rf /tmp/gdb-build

# Install dev deps
RUN apt-get -y update \
    && xargs -a /opt/tt_metal_infra/scripts/docker/requirements_dev.txt apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*
