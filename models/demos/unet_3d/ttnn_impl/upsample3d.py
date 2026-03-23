# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn


def upsample3d(x: ttnn.Tensor, scale_factor: int) -> ttnn.Tensor:
    N, D, H, W, C = x.shape
    D_out, H_out, W_out = D * scale_factor, H * scale_factor, W * scale_factor
    x = ttnn.reshape(x, (N * D, H, W, C))
    x0 = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    if x0.buffer_address() != x.buffer_address():
        ttnn.deallocate(x)
    x1 = ttnn.upsample(
        input_tensor=x0,
        scale_factor=scale_factor,
        memory_config=None,
    )
    ttnn.deallocate(x0)
    x2 = ttnn.reshape(x1, (N, D, H_out * W_out, C))
    x3 = ttnn.upsample(
        input_tensor=x2,
        scale_factor=(scale_factor, 1),
        memory_config=None,
    )
    ttnn.deallocate(x2)
    return ttnn.reshape(x3, (N, D_out, H_out, W_out, C))
