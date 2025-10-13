#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

EXALENS_VERSION="0.1.251013+dev.f3eb0e2-cp310-cp310-linux_x86_64"
EXALENS_TAG="${EXALENS_VERSION%%+*}"
EXALENS_WHEEL="ttexalens-${EXALENS_VERSION}.whl"

wget -O ${EXALENS_WHEEL} \
    https://github.com/tenstorrent/tt-exalens/releases/download/${EXALENS_TAG}/${EXALENS_WHEEL} || exit 1
pip install --no-cache-dir ${EXALENS_WHEEL}
rm ${EXALENS_WHEEL}
