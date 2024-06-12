# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import tt_lib as ttl


def SiLU(x):
    xs = ttl.tensor.sigmoid(x)
    xs = ttnn.mul(xs, x)
    return xs
