# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from models.tt_dit.layers.module import Module


class WanRotaryPosEmbed(Module):
    """Compute RoPE cos/sin for Wan-style 3D positional embedding (frame, height, width).

    Frequencies are computed on host in float64 (matching the reference WanRotaryPosEmbed)
    and returned as device tensors with the **interleaved** layout expected by
    ``wan_fused_rmsnorm_post_allgather``'s ``trans_mat`` pairing of adjacent dimensions:
    ``[c0, c0, c1, c1, ..., c63, c63]``.
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

        self.f_freqs_base = 1.0 / (theta ** (torch.arange(0, self.f_dim, 2)[: self.f_dim // 2].double() / self.f_dim))
        self.h_freqs_base = 1.0 / (theta ** (torch.arange(0, self.h_dim, 2)[: self.h_dim // 2].double() / self.h_dim))
        self.w_freqs_base = 1.0 / (theta ** (torch.arange(0, self.w_dim, 2)[: self.w_dim // 2].double() / self.w_dim))

    def forward(self, grid_ids: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        grid_ids_torch = ttnn.to_torch(ttnn.get_device_tensors(grid_ids)[0]).to(torch.float64)

        f_freqs = grid_ids_torch[:, 0, :].unsqueeze(-1) * self.f_freqs_base
        h_freqs = grid_ids_torch[:, 1, :].unsqueeze(-1) * self.h_freqs_base
        w_freqs = grid_ids_torch[:, 2, :].unsqueeze(-1) * self.w_freqs_base
        freqs = torch.cat([f_freqs, h_freqs, w_freqs], dim=-1).float()

        cos_freqs = freqs.cos()
        sin_freqs = freqs.sin()

        rope_cos = cos_freqs.repeat_interleave(2, dim=-1)  # [B, L, head_dim]
        rope_sin = sin_freqs.repeat_interleave(2, dim=-1)

        rope_cos_tt = ttnn.from_torch(rope_cos, device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        rope_sin_tt = ttnn.from_torch(rope_sin, device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        return rope_cos_tt, rope_sin_tt
