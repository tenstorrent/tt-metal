# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native ttnn port of `rotary_positional_embedding`
(meituan-longcat/LongCat-Video's `dit.blocks.*.attn.rope_3d`, class
`RotaryPositionalEmbedding` in the vendored `longcat_video/modules/rope_3d.py`):

    forward(q, k, grid_size):           # q, k: [B, H, N, head_dim], grid_size=(T,H,W), T*H*W==N
        cos, sin = precompute_freqs_cis_3d(grid_size)   # [N, head_dim], deterministic
        q_ = q*cos + rotate_half(q)*sin
        k_ = k*cos + rotate_half(k)*sin
        return q_, k_

Deterministic (no learned weights) -- cos/sin are computed on host
bit-identically to `precompute_freqs_cis_3d` and uploaded as constants.
`rotate_half` (a fixed signed-permutation of the head_dim axis: interleaved
pairs `(x[2d], x[2d+1]) -> (-x[2d+1], x[2d])`) is expressed as a small
`head_dim x head_dim` constant matmul rather than an on-device
slice/negate/concat -- identical math, same technique used in
`long_cat_single_stream_block`.
"""

from __future__ import annotations

import numpy as np
import torch

import ttnn


def _rotate_half_matrix(head_dim: int) -> torch.Tensor:
    m = np.zeros((head_dim, head_dim), dtype=np.float32)
    idx = np.arange(0, head_dim, 2)
    m[idx, idx + 1] = 1.0
    m[idx + 1, idx] = -1.0
    return torch.from_numpy(m)


def _rope_cos_sin(head_dim: int, grid_size, base: float = 10000.0):
    """Deterministic, shape-only constant table -- computed with numpy (not torch) so this
    never fires a torch/aten host op, regardless of when/how often it's called from the hot
    forward path (see the STRICT TT-ONLY CONTRACT's on-device requirement)."""
    T, H, W = grid_size
    dim_t = head_dim - 4 * (head_dim // 6)
    dim_h = 2 * (head_dim // 6)
    dim_w = 2 * (head_dim // 6)

    def _freqs(dim):
        return 1.0 / (base ** (np.arange(0, dim, 2)[: dim // 2].astype(np.float32) / dim))

    freqs_t = np.einsum("t,f->tf", np.arange(T, dtype=np.float32), _freqs(dim_t))
    freqs_h = np.einsum("h,f->hf", np.arange(H, dtype=np.float32), _freqs(dim_h))
    freqs_w = np.einsum("w,f->wf", np.arange(W, dtype=np.float32), _freqs(dim_w))
    freqs_t = np.repeat(freqs_t, 2, axis=-1)
    freqs_h = np.repeat(freqs_h, 2, axis=-1)
    freqs_w = np.repeat(freqs_w, 2, axis=-1)

    freqs = np.concatenate(
        [
            np.broadcast_to(freqs_t[:, None, None, :], (T, H, W, dim_t)),
            np.broadcast_to(freqs_h[None, :, None, :], (T, H, W, dim_h)),
            np.broadcast_to(freqs_w[None, None, :, :], (T, H, W, dim_w)),
        ],
        axis=-1,
    ).reshape(T * H * W, head_dim)
    return torch.from_numpy(np.cos(freqs)), torch.from_numpy(np.sin(freqs))


class TtRotaryPositionalEmbedding:
    def __init__(self, device: ttnn.Device, torch_module) -> None:
        self.device = device
        self.dtype = ttnn.bfloat16
        self.head_dim = torch_module.head_dim
        self.rope_matrix = ttnn.from_torch(
            _rotate_half_matrix(self.head_dim),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self._rope_cache = {}

    def _rope(self, grid_size):
        key = tuple(grid_size)
        if key not in self._rope_cache:
            cos, sin = _rope_cos_sin(self.head_dim, grid_size)
            cos_tt = ttnn.from_torch(
                cos.reshape(1, 1, *cos.shape),
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            sin_tt = ttnn.from_torch(
                sin.reshape(1, 1, *sin.shape),
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            self._rope_cache[key] = (cos_tt, sin_tt)
        return self._rope_cache[key]

    def _apply(self, t, cos, sin):
        rotated = ttnn.linear(t, self.rope_matrix)
        return t * cos + rotated * sin

    def __call__(self, q: ttnn.Tensor, k: torch.Tensor, grid_size):
        cos, sin = self._rope(grid_size)
        k_tt = ttnn.from_torch(k.to(torch.bfloat16), dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
        q_out = self._apply(q, cos, sin)
        k_out = self._apply(k_tt, cos, sin)
        return q_out, k_out


def build(device: ttnn.Device, torch_module) -> TtRotaryPositionalEmbedding:
    return TtRotaryPositionalEmbedding(device, torch_module)
