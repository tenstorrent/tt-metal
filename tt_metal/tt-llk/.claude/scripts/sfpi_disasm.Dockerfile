# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# The image sfpi_disasm_docker.sh runs in.
#
# sfpi 7.9x links against glibc 2.34, so the comparison needs Ubuntu 22.04 even
# when the host is older. Nothing else is installed: the Python packages come
# from the host through PYTHONPATH, and the toolchain is downloaded per tree.
#
#   docker build -f sfpi_disasm.Dockerfile -t llk-disasm:22.04 .

FROM ubuntu:22.04

RUN apt-get -qq update \
    && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        python3 \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*
