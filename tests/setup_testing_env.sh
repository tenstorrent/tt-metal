#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

SFPI_RELEASE_URL="https://github.com/tenstorrent/sfpi/releases/download/v6.8.0/sfpi-release.tgz"

if [ ! -d "sfpi" ]; then
    echo "sfpi directory not found. Downloading and extracting SFPI release..."
    wget "$SFPI_RELEASE_URL" -O sfpi-release.tgz
    tar -xzf sfpi-release.tgz
    rm sfpi-release.tgz
else
    echo "sfpi directory already exists. Skipping download and extraction."
fi
