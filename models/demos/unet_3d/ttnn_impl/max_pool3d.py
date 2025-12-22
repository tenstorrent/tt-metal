# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def max_pool3d(x: ttnn.Tensor, kernel_size: int, stride: int, padding: int) -> ttnn.Tensor:
    N, D, H, W, C = x.shape
    x = ttnn.reshape(x, (N * D, H, W, C))
    x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
    output_reshaped = ttnn.max_pool2d(
        input_tensor=x,
        batch_size=N * D,
        input_h=H,
        input_w=W,
        channels=C,
        kernel_size=[kernel_size, kernel_size],
        stride=[stride, stride],
        padding=[padding, padding],
        dilation=[1, 1],
        ceil_mode=False,
        memory_config=None,
        # applied_shard_scheme=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        deallocate_input=False,
        reallocate_halo_output=True,
        dtype=x.dtype,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    output_reshaped = ttnn.reshape(output_reshaped, (N, D, H_out * W_out, C))
    out = ttnn.to_layout(output_reshaped, ttnn.ROW_MAJOR_LAYOUT)
    out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.max_pool2d(
        input_tensor=out,
        batch_size=N,
        input_h=D,
        input_w=H_out * W_out,
        channels=C,
        kernel_size=[kernel_size, 1],
        stride=[stride, 1],
        padding=[padding, 0],
        dilation=[1, 1],
        ceil_mode=False,
        memory_config=None,
        # applied_shard_scheme=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        deallocate_input=True,
        reallocate_halo_output=True,
        dtype=x.dtype,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    D_out = (D + 2 * padding - kernel_size) // stride + 1
    out = ttnn.reshape(out, (N, D_out, H_out, W_out, C))
    return out
