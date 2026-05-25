# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Device-side Swin-L transformer building blocks.

Adapted from models/experimental/swin_s/tt/ for Swin-L (ED-Pose config):
  dim=[192, 384, 768, 1536], depths=[2, 2, 18, 2], heads=[6, 12, 24, 48]
  window_size=12, mlp_ratio=4.0

Data format between blocks: (B, H*W, C) in TILE layout.
Inside window attention: converts to RM for reshape/permute, TILE for compute.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


def _roll_device(tensor, shifts, dims):
    """Cyclic shift on device via slice + concat (equivalent to torch.roll)."""
    if isinstance(shifts, int):
        shifts = (shifts,)
    if isinstance(dims, int):
        dims = (dims,)
    result = tensor
    shape = list(result.shape)
    ndim = len(shape)
    for shift, dim in zip(shifts, dims):
        shift = shift % shape[dim]
        if shift == 0:
            continue
        start_l = [0] * ndim
        end_l = list(shape)
        start_r = [0] * ndim
        end_r = list(shape)
        start_l[dim] = shape[dim] - shift
        end_r[dim] = shape[dim] - shift
        left = ttnn.slice(result, start_l, end_l, [1] * ndim)
        right = ttnn.slice(result, start_r, end_r, [1] * ndim)
        result = ttnn.concat([left, right], dim)
    return result


def _compute_shift_mask_cpu(pad_H, pad_W, window_size, shift_size):
    """Compute attention mask for shifted window attention (CPU).

    Returns: (num_windows, 1, ws*ws, ws*ws) float tensor.
    Masked positions = -100.0, unmasked = 0.0.
    """
    ws = window_size
    img_mask = torch.zeros(1, pad_H, pad_W, 1)
    h_slices = (slice(0, -ws), slice(-ws, -shift_size), slice(-shift_size, None))
    w_slices = (slice(0, -ws), slice(-ws, -shift_size), slice(-shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    nH = pad_H // ws
    nW = pad_W // ws
    mask = img_mask.view(1, nH, ws, nW, ws, 1)
    mask = mask.permute(0, 1, 3, 2, 4, 5).contiguous().view(nH * nW, ws * ws)
    attn_mask = mask.unsqueeze(1) - mask.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
    return attn_mask.unsqueeze(1)  # (nW, 1, ws*ws, ws*ws)


def _compute_rel_pos_index(window_size):
    """Compute relative position index for window attention."""
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
    coords_flatten = coords.view(2, -1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += window_size - 1
    relative_coords[:, :, 1] += window_size - 1
    relative_coords[:, :, 0] *= 2 * window_size - 1
    return relative_coords.sum(-1)


HIFI4_CONFIG = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4)


class TTShiftedWindowAttention(LightweightModule):
    """Shifted window multi-head self-attention on device."""

    def __init__(self, device, state_dict, prefix, dim, num_heads, window_size, shift_size):
        super().__init__()
        self.device = device
        self.dim = dim
        self.num_heads = num_heads
        self.ws = window_size
        self.shift_size = shift_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        p = f"{prefix}." if prefix else ""

        self.qkv_w = self._weight(state_dict[f"{p}qkv.weight"], dim, 3 * dim)
        self.qkv_b = self._bias(state_dict[f"{p}qkv.bias"])
        self.proj_w = self._weight(state_dict[f"{p}proj.weight"], dim, dim)
        self.proj_b = self._bias(state_dict[f"{p}proj.bias"])

        table = state_dict[f"{p}relative_position_bias_table"]
        index_key = f"{p}relative_position_index"
        if index_key in state_dict:
            index = state_dict[index_key].long()
        else:
            index = _compute_rel_pos_index(window_size)
        ws_sq = window_size * window_size
        bias = table[index.view(-1)].view(ws_sq, ws_sq, -1).permute(2, 0, 1).contiguous()
        self.rel_pos_bias = ttnn.from_torch(
            bias.unsqueeze(0).to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _weight(self, w, in_dim, out_dim):
        return ttnn.from_torch(
            w.T.contiguous().to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _bias(self, b):
        return ttnn.from_torch(
            b.unsqueeze(0).to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _get_shift_mask(self, pad_H, pad_W):
        mask = _compute_shift_mask_cpu(pad_H, pad_W, self.ws, self.shift_size)
        return ttnn.from_torch(
            mask.to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(self, x, H, W):
        """
        Args:
            x: (B, H, W, C) ttnn in ROW_MAJOR layout
        Returns:
            (B, H, W, C) ttnn in ROW_MAJOR layout
        """
        B = x.shape[0]
        C = self.dim
        ws = self.ws

        pad_b = (ws - H % ws) % ws
        pad_r = (ws - W % ws) % ws
        if pad_b > 0 or pad_r > 0:
            old = x
            x = ttnn.pad(x, (B, H + pad_b, W + pad_r, C), [0, 0, 0, 0], 0)
            ttnn.deallocate(old)
        pad_H = H + pad_b
        pad_W = W + pad_r

        shift = self.shift_size
        if ws >= pad_H or ws >= pad_W:
            shift = 0

        if shift > 0:
            old = x
            x = _roll_device(x, (-shift, -shift), (1, 2))
            ttnn.deallocate(old)

        nH = pad_H // ws
        nW_dim = pad_W // ws
        num_win = nH * nW_dim

        old = x
        x = ttnn.reshape(x, (B, nH, ws, nW_dim, ws, C))
        x = ttnn.permute(x, (0, 1, 3, 2, 4, 5))
        x = ttnn.reshape(x, (B * num_win, ws * ws, C))
        ttnn.deallocate(old)

        old = x
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        ttnn.deallocate(old)
        qkv = ttnn.linear(x, self.qkv_w, bias=self.qkv_b, compute_kernel_config=HIFI4_CONFIG)
        ttnn.deallocate(x)

        ws_sq = ws * ws
        old = qkv
        qkv = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(old)
        q = qkv[:, :, :C]
        k = qkv[:, :, C : 2 * C]
        v = qkv[:, :, 2 * C :]
        ttnn.deallocate(qkv)

        q = ttnn.reshape(q, (B * num_win, ws_sq, self.num_heads, self.head_dim))
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.reshape(k, (B * num_win, ws_sq, self.num_heads, self.head_dim))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.reshape(v, (B * num_win, ws_sq, self.num_heads, self.head_dim))
        v = ttnn.permute(v, (0, 2, 1, 3))

        old = q
        q = q * self.scale
        ttnn.deallocate(old)
        k_t = ttnn.permute(k, (0, 1, 3, 2))
        ttnn.deallocate(k)

        old = q
        q = ttnn.to_layout(q, ttnn.TILE_LAYOUT)
        ttnn.deallocate(old)
        old = k_t
        k_t = ttnn.to_layout(k_t, ttnn.TILE_LAYOUT)
        ttnn.deallocate(old)
        attn = ttnn.matmul(q, k_t, compute_kernel_config=HIFI4_CONFIG)
        ttnn.deallocate(q)
        ttnn.deallocate(k_t)

        old = attn
        attn = ttnn.add(attn, self.rel_pos_bias)
        ttnn.deallocate(old)
        if shift > 0:
            shift_mask = self._get_shift_mask(pad_H, pad_W)
            old = attn
            attn = ttnn.add(attn, shift_mask)
            ttnn.deallocate(old)
            ttnn.deallocate(shift_mask)

        old = attn
        attn = ttnn.softmax(attn, dim=-1)
        ttnn.deallocate(old)

        old = v
        v = ttnn.to_layout(v, ttnn.TILE_LAYOUT)
        ttnn.deallocate(old)
        out = ttnn.matmul(attn, v, compute_kernel_config=HIFI4_CONFIG)
        ttnn.deallocate(attn)
        ttnn.deallocate(v)

        old = out
        out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(old)
        out = ttnn.permute(out, (0, 2, 1, 3))
        out = ttnn.reshape(out, (B * num_win, ws_sq, C))

        old = out
        out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        ttnn.deallocate(old)
        old = out
        out = ttnn.linear(out, self.proj_w, bias=self.proj_b, compute_kernel_config=HIFI4_CONFIG)
        ttnn.deallocate(old)

        old = out
        out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(old)
        out = ttnn.reshape(out, (B, nH, nW_dim, ws, ws, C))
        out = ttnn.permute(out, (0, 1, 3, 2, 4, 5))
        out = ttnn.reshape(out, (B, pad_H, pad_W, C))

        if shift > 0:
            old = out
            out = _roll_device(out, (shift, shift), (1, 2))
            ttnn.deallocate(old)

        if pad_b > 0 or pad_r > 0:
            old = out
            out = out[:, :H, :W, :]
            ttnn.deallocate(old)

        return out


class TTSwinBlock(LightweightModule):
    """One Swin transformer block: LN → WindowAttn → Res → LN → MLP → Res.

    Input/output: (B, H*W, C) in TILE layout.
    """

    def __init__(self, device, state_dict, prefix, dim, num_heads, window_size, shift_size, mlp_ratio=4.0):
        super().__init__()
        self.device = device
        self.dim = dim
        p = f"{prefix}." if prefix else ""

        self.attn = TTShiftedWindowAttention(
            device,
            state_dict,
            f"{p}attn",
            dim,
            num_heads,
            window_size,
            shift_size,
        )

        self.norm1_w = self._ln_param(state_dict[f"{p}norm1.weight"])
        self.norm1_b = self._ln_param(state_dict[f"{p}norm1.bias"])
        self.norm2_w = self._ln_param(state_dict[f"{p}norm2.weight"])
        self.norm2_b = self._ln_param(state_dict[f"{p}norm2.bias"])

        self.fc1_w = self._weight(state_dict[f"{p}mlp.fc1.weight"])
        self.fc1_b = self._bias(state_dict[f"{p}mlp.fc1.bias"])
        self.fc2_w = self._weight(state_dict[f"{p}mlp.fc2.weight"])
        self.fc2_b = self._bias(state_dict[f"{p}mlp.fc2.bias"])

    def _ln_param(self, t):
        return ttnn.from_torch(
            t.unsqueeze(0).to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _weight(self, w):
        return ttnn.from_torch(
            w.T.contiguous().to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _bias(self, b):
        return ttnn.from_torch(
            b.unsqueeze(0).to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(self, x, H, W):
        """
        Args:
            x: (B, H*W, C) ttnn in TILE layout
        Returns:
            (B, H*W, C) ttnn in TILE layout
        """
        B = x.shape[0]
        C = self.dim

        shortcut = ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        old = x
        x = ttnn.layer_norm(x, weight=self.norm1_w, bias=self.norm1_b)
        ttnn.deallocate(old)
        old = x
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(old)
        x = ttnn.reshape(x, (B, H, W, C))
        x = self.attn(x, H, W)
        x = ttnn.reshape(x, (B, H * W, C))
        old = x
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        ttnn.deallocate(old)
        attn_out = x
        x = ttnn.add(shortcut, attn_out)
        ttnn.deallocate(shortcut)
        ttnn.deallocate(attn_out)

        shortcut = ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        old = x
        x = ttnn.layer_norm(x, weight=self.norm2_w, bias=self.norm2_b)
        ttnn.deallocate(old)
        old = x
        x = ttnn.linear(x, self.fc1_w, bias=self.fc1_b, compute_kernel_config=HIFI4_CONFIG)
        ttnn.deallocate(old)
        old = x
        x = ttnn.gelu(x)
        ttnn.deallocate(old)
        old = x
        x = ttnn.linear(x, self.fc2_w, bias=self.fc2_b, compute_kernel_config=HIFI4_CONFIG)
        ttnn.deallocate(old)
        mlp_out = x
        x = ttnn.add(shortcut, mlp_out)
        ttnn.deallocate(shortcut)
        ttnn.deallocate(mlp_out)

        return x


class TTPatchMerging(LightweightModule):
    """Stride-2 downsample: 4x slice → concat → LayerNorm → Linear.

    Input: (B, H*W, C) TILE. Output: (B, H/2*W/2, 2*C) TILE.
    """

    def __init__(self, device, state_dict, prefix, dim):
        super().__init__()
        self.device = device
        self.dim = dim
        p = f"{prefix}." if prefix else ""

        self.norm_w = ttnn.from_torch(
            state_dict[f"{p}norm.weight"].unsqueeze(0).to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.norm_b = ttnn.from_torch(
            state_dict[f"{p}norm.bias"].unsqueeze(0).to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Linear(4*dim, 2*dim, bias=False): weight shape (2*dim, 4*dim)
        self.reduction_w = ttnn.from_torch(
            state_dict[f"{p}reduction.weight"].T.contiguous().to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(self, x, H, W):
        """
        Args:
            x: (B, H*W, C) ttnn in TILE layout
            H, W: spatial dimensions
        Returns:
            (output, new_H, new_W) — output is (B, new_H*new_W, 2*C) TILE
        """
        B = x.shape[0]
        C = self.dim

        old = x
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(old)
        x = ttnn.reshape(x, (B, H, W, C))

        if H % 2 == 1 or W % 2 == 1:
            pad_H = H + (H % 2)
            pad_W = W + (W % 2)
            old = x
            x = ttnn.pad(x, (B, pad_H, pad_W, C), [0, 0, 0, 0], 0)
            ttnn.deallocate(old)
        else:
            pad_H, pad_W = H, W

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        ttnn.deallocate(x)
        x = ttnn.concat([x0, x1, x2, x3], -1)
        ttnn.deallocate(x0)
        ttnn.deallocate(x1)
        ttnn.deallocate(x2)
        ttnn.deallocate(x3)

        new_H = pad_H // 2
        new_W = pad_W // 2
        x = ttnn.reshape(x, (B, new_H * new_W, 4 * C))

        old = x
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        ttnn.deallocate(old)
        old = x
        x = ttnn.layer_norm(x, weight=self.norm_w, bias=self.norm_b)
        ttnn.deallocate(old)
        old = x
        x = ttnn.linear(x, self.reduction_w, compute_kernel_config=HIFI4_CONFIG)
        ttnn.deallocate(old)

        return x, new_H, new_W
