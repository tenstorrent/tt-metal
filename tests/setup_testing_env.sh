#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e  # Exit immediately if a command exits with a non-zero status
set -o pipefail  # Fail if any command in a pipeline fails

SFPI_VERSION=$(cat sfpi_version.txt)

SFPI_RELEASE_URL="https://github.com/tenstorrent/sfpi/releases/download/${SFPI_VERSION}/sfpi-x86_64_Linux.txz"

if [ ! -d "sfpi" ]; then
    echo "sfpi directory not found. Downloading and extracting SFPI ${SFPI_VERSION} release..."
    if ! wget --waitretry=5 --retry-connrefused "$SFPI_RELEASE_URL" -O sfpi-release.txz; then
        echo "Error: Failed to download SFPI release from $SFPI_RELEASE_URL" >&2
        exit 1
    fi

    if ! tar -xJf sfpi-release.txz; then
        echo "Error: Failed to extract SFPI release" >&2
        exit 1
    fi

    rm sfpi-release.txz
else
    echo "sfpi directory already exists. Skipping download and extraction."
fi
