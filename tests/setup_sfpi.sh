#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

SFPI_RELEASE_URL="https://github.com/tenstorrent/sfpi/releases/download/v6.7.0/sfpi-release.tgz"
wget "$SFPI_RELEASE_URL" -O sfpi-release.tgz
tar -xzf sfpi-release.tgz
rm sfpi-release.tgz
