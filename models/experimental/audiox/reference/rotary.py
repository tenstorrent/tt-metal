# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        base: int = 10000,
        base_rescale_factor: float = 1.0,
        interpolation_factor: float = 1.0,
    ):
        super().__init__()
        base *= base_rescale_factor ** (dim / (dim - 2))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        assert interpolation_factor >= 1.0
        self.interpolation_factor = interpolation_factor

    def forward_from_seq_len(self, seq_len: int):
        return self.forward(torch.arange(seq_len, device=self.inv_freq.device))

    def forward(self, t: torch.Tensor):
        t = t.to(torch.float32) / self.interpolation_factor
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs, 1.0


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t: torch.Tensor, freqs: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    rot_dim, seq_len = freqs.shape[-1], t.shape[-2]
    out_dtype = t.dtype
    freqs = freqs.to(torch.float32)
    t_f = t.to(torch.float32)
    freqs = freqs[-seq_len:, :]

    t_rot, t_pass = t_f[..., :rot_dim], t_f[..., rot_dim:]
    t_rot = (t_rot * freqs.cos() * scale) + (rotate_half(t_rot) * freqs.sin() * scale)

    return torch.cat((t_rot.to(out_dtype), t_pass.to(out_dtype)), dim=-1)
