# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.tt_dit.layers.module import Module


def _inv_freq_half(mesh_device: ttnn.MeshDevice, dim: int, theta: float) -> ttnn.Tensor:
    """``1 / (theta ** (i / dim))`` for ``i`` in ``0, 2, …``; length ``dim // 2``. Device float32, TILE."""
    n_half = dim // 2
    indices = ttnn.arange(0, dim, 2, device=mesh_device, dtype=ttnn.float32)
    indices = ttnn.to_layout(indices, ttnn.TILE_LAYOUT)
    if int(indices.shape[0]) > n_half:
        indices = ttnn.slice(indices, [0], [n_half])
    exponents = ttnn.multiply(indices, 1.0 / float(dim))
    ttnn.deallocate(indices)
    base_powers = ttnn.pow(theta, exponents)
    ttnn.deallocate(exponents)
    inv = ttnn.reciprocal(base_powers)
    ttnn.deallocate(base_powers)
    return inv


def _repeat_interleave_pairs_last_dim(x: ttnn.Tensor) -> ttnn.Tensor:
    """``(B, L, H)`` → ``(B, L, 2*H)`` with order ``[x0,x0,x1,x1,...]`` (matches ``torch.repeat_interleave(..., 2, dim=-1)``).

    ``ttnn.concat([x, x], dim=-1)`` would yield ``[x0,…,x_{H-1},x0,…]`` which is wrong for Wan RoPE.
    """
    b, ell, h = (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]))
    x4 = ttnn.reshape(x, (b, ell, h, 1))
    paired = ttnn.concat([x4, x4], dim=-1)
    return ttnn.reshape(paired, (b, ell, h * 2))


class WanRotaryPosEmbed(Module):
    """Compute RoPE cos/sin for Wan-style 3D positional embedding (frame, height, width).

    All math uses **ttnn** on the mesh (float32 for trig, then bf16 outputs). ``grid_ids`` are
    multiplied by inverse-frequency vectors (same recipe as the reference, which used host float64).

    Outputs use the **interleaved** layout expected by ``wan_fused_rmsnorm_post_allgather`` /
    ``trans_mat``: ``[c0, c0, c1, c1, …]`` along the last dimension.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        attention_head_dim: int,
        patch_size,
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.f_dim = self.attention_head_dim - 2 * (self.attention_head_dim // 3)
        self.h_dim = self.attention_head_dim // 3
        self.w_dim = self.attention_head_dim // 3

        self._inv_f = _inv_freq_half(mesh_device, self.f_dim, theta)
        self._inv_h = _inv_freq_half(mesh_device, self.h_dim, theta)
        self._inv_w = _inv_freq_half(mesh_device, self.w_dim, theta)

    def forward(self, grid_ids: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        if grid_ids.layout != ttnn.TILE_LAYOUT:
            grid_ids = ttnn.to_layout(grid_ids, ttnn.TILE_LAYOUT)
        g = ttnn.typecast(grid_ids, ttnn.float32)

        b = int(g.shape[0])
        ell = int(g.shape[2])
        nf, nh, nw = self.f_dim // 2, self.h_dim // 2, self.w_dim // 2

        f_plane = ttnn.squeeze(ttnn.slice(g, [0, 0, 0], [b, 1, ell]), 1)
        h_plane = ttnn.squeeze(ttnn.slice(g, [0, 1, 0], [b, 2, ell]), 1)
        w_plane = ttnn.squeeze(ttnn.slice(g, [0, 2, 0], [b, 3, ell]), 1)

        f_exp = ttnn.reshape(f_plane, (b, ell, 1))
        h_exp = ttnn.reshape(h_plane, (b, ell, 1))
        w_exp = ttnn.reshape(w_plane, (b, ell, 1))

        inv_f = ttnn.reshape(self._inv_f, (1, 1, nf))
        inv_h = ttnn.reshape(self._inv_h, (1, 1, nh))
        inv_w = ttnn.reshape(self._inv_w, (1, 1, nw))

        f_freqs = ttnn.multiply(f_exp, inv_f)
        h_freqs = ttnn.multiply(h_exp, inv_h)
        w_freqs = ttnn.multiply(w_exp, inv_w)

        freqs = ttnn.concat([f_freqs, h_freqs, w_freqs], dim=-1)
        cos_h = ttnn.cos(freqs)
        sin_h = ttnn.sin(freqs)

        rope_cos = _repeat_interleave_pairs_last_dim(cos_h)
        rope_sin = _repeat_interleave_pairs_last_dim(sin_h)

        rope_cos = ttnn.typecast(rope_cos, ttnn.bfloat16)
        rope_sin = ttnn.typecast(rope_sin, ttnn.bfloat16)
        return rope_cos, rope_sin
