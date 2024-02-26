# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


def upsample_nearest2d(input, scale_factor=2.0):
    assert scale_factor % 1 == 0 and scale_factor > 0, "We only support scaling by positive integer values"

    # up_output = ttnn.repeat_interleave(input, scale_factor, dim=3)
    # up_output = ttnn.repeat_interleave(up_output, scale_factor, dim=2)

    ## permute to NHWC
    up_output = ttnn.upsample(input, scale_factor)
    return up_output
