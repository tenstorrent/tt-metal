# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def upsample3d(x: ttnn.Tensor, scale_factor: int) -> ttnn.Tensor:
    """
    3D upsampling implemented using two 2D upsampling operations.
    It is done by first reshaping to combine the depth and batch dimensions,
    then performing 2D upsampling on height and width, reshaping back,
    and finally performing 2D upsampling on depth and the combined height-width dimension.
    """
    N, D, H, W, C = x.shape
    x = ttnn.reshape(x, (N * D, H, W, C))
    x0 = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x1 = ttnn.upsample(
        input_tensor=x0,
        scale_factor=scale_factor,
        memory_config=None,
    )
    H_out = H * scale_factor
    W_out = W * scale_factor
    x1 = ttnn.reshape(x1, (N, D, H_out * W_out, C))
    x2 = ttnn.upsample(
        input_tensor=x1,
        scale_factor=(scale_factor, 1),
        memory_config=None,
    )
    D_out = D * scale_factor

    x2 = ttnn.reshape(x2, (N, D_out, H_out, W_out, C))
    return x2
