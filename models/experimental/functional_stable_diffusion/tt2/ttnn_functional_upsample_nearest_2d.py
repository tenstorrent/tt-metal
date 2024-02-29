# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


def upsample_nearest2d(input, scale_factor=2.0):
    assert scale_factor % 1 == 0 and scale_factor > 0, "We only support scaling by positive integer values"

    # up_output = ttnn.repeat_interleave(input, scale_factor, dim=3)
    # up_output = ttnn.repeat_interleave(up_output, scale_factor, dim=2)

    ## permute to NHWC
    input = ttnn.to_layout(input, ttnn.ROW_MAJOR_LAYOUT)
    input = ttnn.permute(input, (0, 2, 3, 1))

    up_output = ttnn.upsample(input, scale_factor)

    ## permute back to NCHW
    up_output = ttnn.permute(up_output, (0, 3, 1, 2))

    return up_output
