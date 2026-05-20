# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Shifted Window Attention for Swin-L backbone.

Uses fused scaled_dot_product_attention kernel to replace manual
matmul(Q, K^T) + softmax + matmul(attn, V) with a single kernel call.
"""

import torch
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
    """Shifted window multi-head self-attention using fused SDPA kernel."""

    def __init__(self, device, parameters, dim, window_size, shift_size, num_heads, attn_mask=None):
        self.device = device
        self.parameters = parameters
        self.dim = dim
        self.window_size = list(window_size) if not isinstance(window_size, list) else window_size
        self.shift_size = list(shift_size) if not isinstance(shift_size, list) else shift_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_mask = attn_mask
        self._sdpa_mask = None
        self._compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
            math_approx_mode=False,
        )

    def _prepare_sdpa_mask(self, num_windows):
        """Precompute combined mask (relative position bias + window mask) for fused SDPA."""
        rpb_tt = self.parameters["relative_position_bias"]

        has_shift = sum(self.shift_size) > 0
        if self.attn_mask is not None and has_shift:
            rpb = ttnn.to_torch(ttnn.from_device(rpb_tt)).float()
            combined = rpb + self.attn_mask.float()
            self._sdpa_mask = ttnn.from_torch(
                combined.to(torch.bfloat16),
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self._sdpa_mask = rpb_tt

    def __call__(self, input_tensor):
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

        qkv = ttnn.linear(
            ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            self.parameters["qkv"]["weight"],
            bias=self.parameters["qkv"]["bias"],
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, packer_l1_acc=True
            ),
            core_grid=ttnn.CoreGrid(y=8, x=8),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(input_tensor)

        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, num_heads=self.num_heads, transpose_key=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(qkv)

        if self._sdpa_mask is None:
            self._prepare_sdpa_mask(num_windows)

        output = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=self._sdpa_mask,
            is_causal=False,
            scale=self.head_dim**-0.5,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                q_chunk_size=32,
                k_chunk_size=32,
            ),
            compute_kernel_config=self._compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        output = ttnn.transformer.concatenate_heads(output, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        output = ttnn.linear(
            output,
            self.parameters["proj"]["weight"],
            bias=self.parameters["proj"]["bias"],
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False
            ),
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
