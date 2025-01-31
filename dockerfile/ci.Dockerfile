
# Accept an argument to specify the Ubuntu version
ARG UBUNTU_VERSION=20.04
ARG TAG=latest
FROM ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-${UBUNTU_VERSION}-amd64-base:${TAG}

# Install build deps
# Script is already there from base layer
# COPY /install_dependencies.sh /opt/tt_metal_infra/scripts/docker/install_dependencies.sh
RUN /bin/bash /opt/tt_metal_infra/scripts/docker/install_dependencies.sh --docker --mode build
