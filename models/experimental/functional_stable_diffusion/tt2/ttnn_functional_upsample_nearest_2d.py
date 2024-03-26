# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


def upsample_nearest2d(input, scale_factor=2.0):
    assert scale_factor % 1 == 0 and scale_factor > 0, "We only support scaling by positive integer values"

    # input is in N, 1, HW, C, upsample expects, [N, H, W, C]
    # set h_scale to 1, w_scale to scale_factor, c_scale to 1
    # scale_factor = (1, scale_factor*2, 1)
    up_output = ttnn.upsample(input, scale_factor)
    return up_output
