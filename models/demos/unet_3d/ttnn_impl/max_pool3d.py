# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn


def max_pool3d(x: ttnn.Tensor, kernel_size: int) -> ttnn.Tensor:
    N, D, H, W, C = x.shape
    H_out = H // kernel_size
    W_out = W // kernel_size
    D_out = D // kernel_size
    x0 = ttnn.reshape(x, (1, 1, N * D * H * W, C))
    deallocate_input = x0.buffer_address() != x.buffer_address()
    x1 = ttnn.max_pool2d(
        input_tensor=x0,
        batch_size=N * D,
        input_h=H,
        input_w=W,
        channels=C,
        kernel_size=[kernel_size, kernel_size],
        stride=[kernel_size, kernel_size],
        padding=[0, 0],
        dilation=[1, 1],
        ceil_mode=False,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        deallocate_input=deallocate_input,
        reallocate_halo_output=True,
        dtype=x.dtype,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    if deallocate_input:
        ttnn.deallocate(x0)

    x2 = ttnn.max_pool2d(
        input_tensor=x1,
        batch_size=N,
        input_h=D,
        input_w=H_out * W_out,
        channels=C,
        kernel_size=[kernel_size, 1],
        stride=[kernel_size, 1],
        padding=[0, 0],
        dilation=[1, 1],
        ceil_mode=False,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        deallocate_input=True,
        reallocate_halo_output=True,
        dtype=x.dtype,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    ttnn.deallocate(x1)
    y = ttnn.reshape(x2, (N, D_out, H_out, W_out, C))
    return y
