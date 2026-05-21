# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Shifted Window Attention for Swin-L backbone.
Adapted from models/experimental/swin_s/tt/tt_shifted_window_attention.py.
Initial version: correctness-first (no hardcoded sharding configs).
"""

import ttnn


def roll(tensor, shifts, dims):
    """Cyclic shift — uses native ttnn.roll (single op), with a slice+concat fallback."""
    if isinstance(shifts, int):
        shifts = [shifts]
    if isinstance(dims, int):
        dims = [dims]
    shifts = list(shifts)
    dims = list(dims)

    shape = tensor.shape
    # Drop zero shifts to keep the call minimal.
    filtered = [(s % shape[d], d) for s, d in zip(shifts, dims) if (s % shape[d]) != 0]
    if not filtered:
        return tensor
    nz_shifts = [s for s, _ in filtered]
    nz_dims = [d for _, d in filtered]
    return ttnn.roll(tensor, shifts=nz_shifts, dim=nz_dims)


class TtSwinAttention:
    """Shifted window multi-head self-attention (TTNN)."""

    def __init__(self, device, parameters, dim, window_size, shift_size, num_heads, attn_mask=None):
        import torch

        self.device = device
        self.parameters = dict(parameters)  # shallow copy so we can replace qkv weights
        self.dim = dim
        self.window_size = list(window_size) if not isinstance(window_size, list) else window_size
        self.shift_size = list(shift_size) if not isinstance(shift_size, list) else shift_size
        self.num_heads = num_heads
        self.attn_mask = attn_mask

        # Pre-scale the Q rows of qkv.weight (and qkv.bias) by 1/sqrt(head_dim) so we don't
        # need a per-call `q *= scale` pass on the full (B*nW, H, S, D) tensor.
        head_dim = dim // num_heads
        scale = head_dim**-0.5
        qkv_w = ttnn.to_torch(parameters["qkv"]["weight"]).float()  # (C, 3C) in linear-ready form
        qkv_b = ttnn.to_torch(parameters["qkv"]["bias"]).float()  # (1, 3C)
        qkv_w[:, :dim] *= scale
        qkv_b[:, :dim] *= scale
        self.parameters["qkv"] = {
            "weight": ttnn.from_torch(
                qkv_w.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            ),
            "bias": ttnn.from_torch(
                qkv_b.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            ),
        }

        # For shift!=0 layers, pre-combine rel_pos_bias + window_mask on host into a
        # single (nW, num_heads, N, N) bias so __call__ does a single add (instead of
        # two adds with broadcast). For shift==0 layers we keep rpb as-is.
        rpb = parameters["relative_position_bias"]  # (1, num_heads, N, N) TILE bf16 on device
        if attn_mask is not None and sum(self.shift_size) > 0:
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
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
        )
        ttnn.deallocate(input_tensor)

        # Fused: split QKV + reshape to (B*nW, H, S, D) + transpose K for matmul (Q,K,V all TILE).
        q, k_t, v = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv,
            num_heads=self.num_heads,
            transpose_key=True,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)

        # Q rows of qkv weight are already pre-scaled in __init__, so no runtime scale here.
        # NOTE: tried ttnn.transformer.scaled_dot_product_attention with the 1280-branch's
        # SDPAProgramConfig(q_chunk_size=32, k_chunk_size=32) at 640x640 — PCC dropped to 0.85.
        # Likely cause: at 640 nW=196 (14x14), the SDPA flash chunking precision compounds
        # too much across 24 stacked blocks. The 1280 branch (nW=8100, 90x90) doesn't see
        # this because the aggregated batch dim averages out the per-chunk rounding.
        # Stayed on the manual matmul+softmax path at 640.
        attn = ttnn.matmul(
            q,
            k_t,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, packer_l1_acc=True
            ),
            core_grid=ttnn.CoreGrid(y=8, x=8),
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k_t)

        attn = ttnn.add(attn, self.combined_attn_bias, memory_config=ttnn.L1_MEMORY_CONFIG)
        attn = ttnn.softmax(attn, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)

        output = ttnn.matmul(
            attn,
            v,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False
            ),
            core_grid=ttnn.CoreGrid(y=8, x=8),
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
        )
        ttnn.deallocate(v)
        ttnn.deallocate(attn)

        # Concatenate heads back: (B*nW, H, S, D) -> (B*nW, S, H*D)
        output = ttnn.transformer.concatenate_heads(output, memory_config=ttnn.L1_MEMORY_CONFIG)

        # concat_heads preserves the matmul-attn@V output layout (TILE), so no explicit
        # to_layout is needed before the proj linear — skipping it saves a dispatch.
        output = ttnn.linear(
            output,
            self.parameters["proj"]["weight"],
            bias=self.parameters["proj"]["bias"],
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False
            ),
            core_grid=ttnn.CoreGrid(y=8, x=8),
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
        )

        output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        output = ttnn.reshape(output, (B, nH, nW, wH, wW, C))
        output = ttnn.transpose(output, 2, 3, memory_config=ttnn.L1_MEMORY_CONFIG)
        output = ttnn.reshape(output, (B, pad_H, pad_W, C))

        # reverse cyclic shift
        if sum(shift_size) > 0:
            output = roll(output, (shift_size[0], shift_size[1]), [1, 2])

        # unpad
        if pad_b > 0 or pad_r > 0:
            output = ttnn.slice(output, [0, 0, 0, 0], [B, H, W, C], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return output
