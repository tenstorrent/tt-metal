#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Single source of truth for external dependency versions
# This file should be sourced by installation scripts

# tt-exalens configuration
export EXALENS_VERSION="0.2.2"

# tt-smi configuration
export TT_SMI_VERSION="3.0.38"
export TT_SMI_WHEEL="tt_smi-${TT_SMI_VERSION}-py3-none-any.whl"
export TT_SMI_URL="https://github.com/tenstorrent/tt-smi/releases/download/v${TT_SMI_VERSION}/${TT_SMI_WHEEL}"

export TT_UMD_VERSION="0.7.4"
