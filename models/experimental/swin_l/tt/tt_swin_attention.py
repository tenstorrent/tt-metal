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

        # For shift!=0 layers, pre-combine rel_pos_bias + window_mask on host into a
        # single (nW, num_heads, N, N) bias so __call__ does a single add (instead of
        # two adds with broadcast). For shift==0 layers we keep rpb as-is.
        rpb = parameters["relative_position_bias"]  # (1, num_heads, N, N) TILE bf16 on device
        if attn_mask is not None and sum(self.shift_size) > 0:
            import torch

            rpb_t = ttnn.to_torch(rpb).float()  # (1, H, N, N)
            mask_t = ttnn.to_torch(attn_mask).float()  # 5D, last two dims are (N, N)
            N = mask_t.shape[-1]
            nW = mask_t.numel() // (N * N)
            mask_t = mask_t.reshape(nW, 1, N, N)
            combined = (rpb_t + mask_t).to(torch.bfloat16)  # (nW, H, N, N)
            self.combined_attn_bias = ttnn.from_torch(
                combined, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
        else:
            self.combined_attn_bias = rpb  # (1, H, N, N), broadcasts over batch in plain add

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

        # cyclic shift
        if sum(shift_size) > 0:
            input_tensor = roll(input_tensor, (-shift_size[0], -shift_size[1]), [1, 2])

        # partition windows
        num_windows = (pad_H // self.window_size[0]) * (pad_W // self.window_size[1])
        nH = pad_H // self.window_size[0]
        nW = pad_W // self.window_size[1]
        wH = self.window_size[0]
        wW = self.window_size[1]

        input_tensor = ttnn.reshape(
            ttnn.transpose(ttnn.reshape(input_tensor, (B, nH, wH, nW, wW, C)), 2, 3), (B * num_windows, wH * wW, C)
        )

        # QKV projection
        seq_len = wH * wW
        head_dim = C // self.num_heads

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

        # Fused: split QKV + reshape to (B*nW, H, S, D) + transpose K for matmul (Q,K,V all TILE).
        q, k_t, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv,
            num_heads=self.num_heads,
            transpose_key=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)

        # Pre-scale Q before matmul (no separate scalar mul on the bigger attn tensor).
        scale = head_dim**-0.5
        q = ttnn.multiply(q, scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        attn = ttnn.matmul(
            q,
            k_t,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, packer_l1_acc=True
            ),
            core_grid=ttnn.CoreGrid(y=8, x=8),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k_t)

        # Single add for the combined bias (rel_pos_bias for shift==0, or rel_pos_bias +
        # window_mask pre-combined in __init__ for shift!=0). Replaces the previous
        # two-step add chain.
        attn = ttnn.add(attn, self.combined_attn_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn = ttnn.softmax(attn, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        output = ttnn.matmul(
            attn,
            v,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False
            ),
            core_grid=ttnn.CoreGrid(y=8, x=8),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(v)
        ttnn.deallocate(attn)

        # Concatenate heads back: (B*nW, H, S, D) -> (B*nW, S, H*D)
        output = ttnn.transformer.concatenate_heads(output, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        output = ttnn.linear(
            ttnn.to_layout(output, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG),
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

        # reverse cyclic shift
        if sum(shift_size) > 0:
            output = roll(output, (shift_size[0], shift_size[1]), [1, 2])

        # unpad
        if pad_b > 0 or pad_r > 0:
            output = ttnn.slice(output, [0, 0, 0, 0], [B, H, W, C], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return output
