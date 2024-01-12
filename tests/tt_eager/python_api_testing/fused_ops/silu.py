# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import numpy as np
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")
import tt_lib as ttl


def SiLU(x):
    xs = ttl.tensor.sigmoid(x)
    xs = ttl.tensor.mul(xs, x)
    return xs
