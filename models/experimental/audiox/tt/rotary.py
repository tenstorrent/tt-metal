# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from models.experimental.audiox.tt.common import to_tt


def precompute_rotary_cos_sin(
    seq_len: int,
    dim: int,
    mesh_device,
    base: int = 10000,
    base_rescale_factor: float = 1.0,
    interpolation_factor: float = 1.0,
    dtype=ttnn.bfloat16,
):
    """Precompute rotary cos/sin tables (use_xpos=False path). Returns
    (cos, sin) ttnn tensors of shape [1, 1, seq_len, dim] ready to broadcast
    against a [batch, heads, seq_len, head_dim] activation."""
    base = base * (base_rescale_factor ** (dim / (dim - 2)))
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len, dtype=torch.float32) / interpolation_factor
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    freqs = torch.cat((freqs, freqs), dim=-1)
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)
    return to_tt(cos, mesh_device, dtype=dtype), to_tt(sin, mesh_device, dtype=dtype)


def apply_rotary_pos_emb(t: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
    """Apply rotary embeddings to `t` in [batch, heads, seq, head_dim] layout.
    `cos`/`sin` come from `precompute_rotary_cos_sin` and have shape
    [1, 1, seq, rot_dim]. Supports the partial-rotary case where
    rot_dim <= head_dim; the trailing (head_dim - rot_dim) features pass
    through unchanged."""
    rot_dim = cos.shape[-1]
    head_dim = t.shape[-1]
    batch, heads, seq, _ = t.shape

    if rot_dim == head_dim:
        t_rot = t
        t_pass = None
    else:
        t_rot = ttnn.slice(t, [0, 0, 0, 0], [batch, heads, seq, rot_dim])
        t_pass = ttnn.slice(t, [0, 0, 0, rot_dim], [batch, heads, seq, head_dim])

    half = rot_dim // 2
    x1 = ttnn.slice(t_rot, [0, 0, 0, 0], [batch, heads, seq, half])
    x2 = ttnn.slice(t_rot, [0, 0, 0, half], [batch, heads, seq, rot_dim])
    rotated = ttnn.concat([ttnn.neg(x2), x1], dim=-1)

    out = ttnn.add(ttnn.multiply(t_rot, cos), ttnn.multiply(rotated, sin))

    if t_pass is not None:
        out = ttnn.concat([out, t_pass], dim=-1)
    return out
