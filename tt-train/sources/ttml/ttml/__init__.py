# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys


def setup_ttml_so():
    sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/build/sources/ttml')
    sys.path.append(f'{os.environ["TT_METAL_HOME"]}/build/tt-train/sources/ttml')


setup_ttml_so()
from _ttml import *
