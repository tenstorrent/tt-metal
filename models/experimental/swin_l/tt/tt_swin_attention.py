# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Shifted Window Attention for Swin-L backbone.
Adapted from models/experimental/swin_s/tt/tt_shifted_window_attention.py.
Initial version: correctness-first (no hardcoded sharding configs).
"""

import ttnn


def roll(tensor, shifts, dims):
    """Cyclic shift via slice + concat (same as Swin-S)."""
    if isinstance(shifts, int):
        shifts = (shifts,)
    if isinstance(dims, int):
        dims = (dims,)
    assert len(shifts) == len(dims)
    result = tensor
    shape = result.shape
    num_dims = len(shape)

    for shift, dim in zip(shifts, dims):
        shift %= shape[dim]
        if shift == 0:
            continue
        start_left = [0] * num_dims
        end_left = list(shape)
        start_right = [0] * num_dims
        end_right = list(shape)
        start_left[dim] = shape[dim] - shift
        end_right[dim] = shape[dim] - shift

        left_part = ttnn.slice(
            result,
            slice_start=start_left,
            slice_end=end_left,
            slice_step=[1] * num_dims,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        right_part = ttnn.slice(
            result,
            slice_start=start_right,
            slice_end=end_right,
            slice_step=[1] * num_dims,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        result = ttnn.concat([left_part, right_part], dim, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return result


class TtSwinAttention:
    """Shifted window multi-head self-attention (TTNN)."""

    def __init__(self, device, parameters, dim, window_size, shift_size, num_heads, attn_mask=None):
        self.device = device
        self.parameters = parameters
        self.dim = dim
        self.window_size = list(window_size) if not isinstance(window_size, list) else window_size
        self.shift_size = list(shift_size) if not isinstance(shift_size, list) else shift_size
        self.num_heads = num_heads
        self.attn_mask = attn_mask

    def __call__(self, input_tensor):
        relative_position_bias = self.parameters["relative_position_bias"]

        B, H, W, C = input_tensor.shape
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_values = (B, H + pad_b, W + pad_r, C)
        input_tensor = ttnn.pad(input_tensor, pad_values, [0, 0, 0, 0], 0)
        _, pad_H, pad_W, _ = input_tensor.shape

        shift_size = self.shift_size.copy()
        if self.window_size[0] >= pad_H:
            shift_size[0] = 0
        if self.window_size[1] >= pad_W:
            shift_size[1] = 0

        # cyclic shift
        if sum(shift_size) > 0:
            input_tensor = roll(input_tensor, (-shift_size[0], -shift_size[1]), [1, 2])

        # partition windows
        num_windows = (pad_H // self.window_size[0]) * (pad_W // self.window_size[1])
        input_tensor = ttnn.reshape(
            input_tensor,
            (
                B,
                pad_H // self.window_size[0],
                self.window_size[0],
                pad_W // self.window_size[1],
                self.window_size[1],
                C,
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        input_tensor = ttnn.permute(input_tensor, (0, 1, 3, 2, 4, 5), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        input_tensor = ttnn.reshape(
            input_tensor,
            (B * num_windows, self.window_size[0] * self.window_size[1], C),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # QKV projection
        input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        qkv = ttnn.linear(
            input_tensor,
            self.parameters["qkv"]["weight"],
            bias=self.parameters["qkv"]["bias"],
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        qkv = ttnn.to_layout(qkv, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        seq_len = input_tensor.shape[1]
        head_dim = C // self.num_heads
        qkv = ttnn.reshape(qkv, (B * num_windows, seq_len, 3, self.num_heads, head_dim))
        qkv = ttnn.permute(qkv, (2, 0, 3, 1, 4), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(input_tensor)

        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
        q = ttnn.squeeze(q, 0)
        k = ttnn.squeeze(k, 0)
        v = ttnn.squeeze(v, 0)

        # scaled dot-product attention
        q = q * (head_dim**-0.5)
        k = ttnn.permute(k, (0, 1, 3, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q = ttnn.to_layout(q, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.to_layout(k, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn = ttnn.matmul(
            q,
            k,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)
        ttnn.deallocate(q)
        ttnn.deallocate(k)

        # relative position bias
        attn = ttnn.add(attn, relative_position_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # attention mask for shifted windows
        if sum(shift_size) > 0 and self.attn_mask is not None:
            attn = attn + self.attn_mask
            attn = ttnn.to_layout(attn, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            attn = ttnn.reshape(attn, (-1, self.num_heads, seq_len, seq_len))
            attn = ttnn.to_layout(attn, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        attn = ttnn.softmax(attn, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        v = ttnn.to_layout(v, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.matmul(
            attn,
            v,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(v)
        ttnn.deallocate(attn)

        output = ttnn.permute(output, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.to_layout(output, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.reshape(output, (B * num_windows, seq_len, C))

        # output projection
        output = ttnn.to_layout(output, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.linear(
            output,
            self.parameters["proj"]["weight"],
            bias=self.parameters["proj"]["bias"],
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # reverse windows
        output = ttnn.to_layout(output, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.reshape(
            output,
            (
                B,
                pad_H // self.window_size[0],
                pad_W // self.window_size[1],
                self.window_size[0],
                self.window_size[1],
                C,
            ),
        )
        output = ttnn.permute(output, (0, 1, 3, 2, 4, 5), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.reshape(output, (B, pad_H, pad_W, C))

        # reverse cyclic shift
        if sum(shift_size) > 0:
            output = roll(output, (shift_size[0], shift_size[1]), [1, 2])

        # unpad
        output = output[:, :H, :W, :]
        return output
