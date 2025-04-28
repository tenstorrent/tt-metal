#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

SFPI_VERSION=$(cat sfpi_version.txt)

SFPI_RELEASE_URL="https://github.com/tenstorrent/sfpi/releases/download/${SFPI_VERSION}/sfpi-x86_64-Linux.tgz"

if [ ! -d "sfpi" ]; then
    echo "sfpi directory not found. Downloading and extracting SFPI ${SFPI_VERSION} release..."
    wget "$SFPI_RELEASE_URL" -O sfpi-release.tgz
    tar -xzf sfpi-release.tgz
    rm sfpi-release.tgz
else
    echo "sfpi directory already exists. Skipping download and extraction."
fi
