#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

wget -O ttexalens-0.1.250326+dev.0c4381a-cp310-cp310-linux_x86_64.whl \
    https://github.com/tenstorrent/tt-exalens/releases/download/0.1/ttexalens-0.1.250326+dev.0c4381a-cp310-cp310-linux_x86_64.whl
pip install --no-cache-dir ttexalens-0.1.250326+dev.0c4381a-cp310-cp310-linux_x86_64.whl
rm ttexalens-0.1.250326+dev.0c4381a-cp310-cp310-linux_x86_64.whl
