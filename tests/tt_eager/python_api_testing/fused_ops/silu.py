# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def SiLU(x):
    xs = ttnn.sigmoid(x)
    xs = ttnn.multiply(xs, x)
    return xs
