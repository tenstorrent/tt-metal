#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

EXALENS_VERSION="0.1.250626+dev.7538f60-cp310-cp310-linux_x86_64"
EXALENS_WHEEL="ttexalens-${EXALENS_VERSION}.whl"

wget -O ${EXALENS_WHEEL} \
    https://github.com/tenstorrent/tt-exalens/releases/download/0.1.250626/${EXALENS_WHEEL}
pip install --no-cache-dir ${EXALENS_WHEEL}
rm ${EXALENS_WHEEL}
