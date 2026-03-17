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

        left_part = ttnn.slice(result, slice_start=start_left, slice_end=end_left, slice_step=[1] * num_dims)
        right_part = ttnn.slice(result, slice_start=start_right, slice_end=end_right, slice_step=[1] * num_dims)
        result = ttnn.concat([left_part, right_part], dim)
        ttnn.deallocate(left_part)
        ttnn.deallocate(right_part)
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
        # Pre-combine relative_position_bias + attn_mask for shifted blocks.
        # rpb: (1, heads, seq, seq). attn_mask: (1, nW, 1, seq, seq) or similar.
        # Reshape attn_mask to 4D (nW, 1, seq, seq) so the addition stays 4D.
        if sum(shift_size) > 0 and attn_mask is not None:
            seq = self.window_size[0] * self.window_size[1]
            # attn_mask is [1, nW, 1, seq, seq]
            # We want to reshape it to [nW, 1, seq, seq] to broadcast with [1, heads, seq, seq]
            shape = attn_mask.shape
            nW = int(shape[0]) * int(shape[1]) * int(shape[2])
            attn_mask_4d = ttnn.reshape(attn_mask, (nW, 1, seq, seq))
            combined = ttnn.add(
                parameters["relative_position_bias"],
                attn_mask_4d,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.combined_bias = ttnn.to_layout(combined, ttnn.TILE_LAYOUT)
        else:
            self.combined_bias = None
        self._hifi2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, packer_l1_acc=True
        )

    def __call__(self, input_tensor):
        relative_position_bias = self.parameters["relative_position_bias"]

        B, H, W, C = input_tensor.shape
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_H = H + pad_b
        pad_W = W + pad_r

        if pad_b > 0 or pad_r > 0:
            input_tensor = ttnn.pad(input_tensor, [(0, 0), (0, pad_b), (0, pad_r), (0, 0)], value=0.0)

        shift_size = self.shift_size.copy()
        if self.window_size[0] >= pad_H:
            shift_size[0] = 0
        if self.window_size[1] >= pad_W:
            shift_size[1] = 0

        if sum(shift_size) > 0:
            input_tensor = roll(input_tensor, (-shift_size[0], -shift_size[1]), [1, 2])

        num_windows = (pad_H // self.window_size[0]) * (pad_W // self.window_size[1])
        nH = pad_H // self.window_size[0]
        nW = pad_W // self.window_size[1]
        wH = self.window_size[0]
        wW = self.window_size[1]

        input_tensor = ttnn.reshape(
            ttnn.transpose(ttnn.reshape(input_tensor, (B, nH, wH, nW, wW, C)), 2, 3), (B * num_windows, wH * wW, C)
        )

        seq_len = wH * wW
        head_dim = C // self.num_heads
        _hifi2 = self._hifi2

        qkv = ttnn.linear(
            ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            self.parameters["qkv"]["weight"],
            bias=self.parameters["qkv"]["bias"],
            compute_kernel_config=_hifi2,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(input_tensor)

        qkv = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q = ttnn.slice(qkv, [0, 0, 0], [B * num_windows, seq_len, C])
        k = ttnn.slice(qkv, [0, 0, C], [B * num_windows, seq_len, 2 * C])
        v = ttnn.slice(qkv, [0, 0, 2 * C], [B * num_windows, seq_len, 3 * C])
        ttnn.deallocate(qkv)

        q = ttnn.transpose(ttnn.reshape(q, (B * num_windows, seq_len, self.num_heads, head_dim)), 1, 2)
        k = ttnn.permute(ttnn.reshape(k, (B * num_windows, seq_len, self.num_heads, head_dim)), (0, 2, 3, 1))
        v = ttnn.transpose(ttnn.reshape(v, (B * num_windows, seq_len, self.num_heads, head_dim)), 1, 2)

        scale = head_dim**-0.5
        q = ttnn.to_layout(
            ttnn.multiply(q, scale, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        k = ttnn.to_layout(k, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn = ttnn.matmul(
            q,
            k,
            compute_kernel_config=_hifi2,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)

        if self.combined_bias is not None:
            attn = ttnn.add(attn, self.combined_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            attn = ttnn.add(attn, relative_position_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        attn = ttnn.softmax(attn, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        v = ttnn.to_layout(v, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.matmul(
            attn,
            v,
            compute_kernel_config=_hifi2,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(v)
        ttnn.deallocate(attn)

        output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.transpose(output, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.reshape(output, (B * num_windows, seq_len, C))

        output = ttnn.linear(
            ttnn.to_layout(output, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            self.parameters["proj"]["weight"],
            bias=self.parameters["proj"]["bias"],
            compute_kernel_config=_hifi2,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.reshape(output, (B, nH, nW, wH, wW, C))
        output = ttnn.transpose(output, 2, 3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.reshape(output, (B, pad_H, pad_W, C))

        if sum(shift_size) > 0:
            output = roll(output, (shift_size[0], shift_size[1]), [1, 2])

        if pad_b > 0 or pad_r > 0:
            output = ttnn.slice(output, [0, 0, 0, 0], [B, H, W, C], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return output
