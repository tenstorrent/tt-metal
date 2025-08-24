#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

TT_SMI_VERSION="3.0.28"
TT_SMI_WHEEL="tt_smi-${TT_SMI_VERSION}-py3-none-any.whl"

wget -O ${TT_SMI_WHEEL} \
    https://github.com/tenstorrent/tt-smi/releases/download/v${TT_SMI_VERSION}/${TT_SMI_WHEEL} || exit 1
pip install --no-cache-dir ${TT_SMI_WHEEL}
rm ${TT_SMI_WHEEL}
