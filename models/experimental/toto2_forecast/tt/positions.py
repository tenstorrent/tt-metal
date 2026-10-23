# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Toto 2.0 partial RoPE + xPos tables (pure torch), time layers only.

reference : toto2/toto2/model.py @ 9213d2c, ``RotaryProjection`` (293-341), ``ExtrapolatableRotaryProjection``
(344-386), ``QueryKeyProjection`` with ``partial_factor`` (0.0, 0.5)
    dims 0-31 of each 64-dim head, interleaved pairs (2i, 2i+1), theta_i = base^(-2i/32):
        rope(x) = cos * x + sin * rot(x),  rot(x)_{2i} = -x_{2i+1},  rot(x)_{2i+1} = x_{2i}
    xPos on the same dims: zeta_i = (2i + 0.4 * 32) / (1.4 * 32),  power = (pos - floor(S / 2)) / 256
        q <- rope(q) * zeta^power,  k <- rope(k) * zeta^-power     (the centre depends on S)
    dims 32-63 pass through

The xPos factor is folded into the tables: ``q' = q * cos_q + rot(q) * sin_q`` with ``cos_q = cos * zeta^p``,
``sin_q = sin * zeta^p`` (``k``: ``zeta^-p``), and cos = 1, sin = 0 on dims 32-63, the layout
``ttnn.experimental.rotary_embedding_llama`` reads (``rot`` per 32-wide tile).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RopeTables:
    """``(S, head_dim)`` float32 tables for the queries and the keys."""

    cos_q: Tensor
    sin_q: Tensor
    cos_k: Tensor
    sin_k: Tensor


def rope_xpos_tables(
    seq_len: int, *, head_dim: int = 64, rope_dims: int = 32, base: float = 10000.0, xpos_scale_base: float = 256.0
) -> RopeTables:
    """Tables for positions ``0 .. seq_len - 1`` (xPos centre ``seq_len // 2``), float32 as the reference."""
    even = torch.arange(0, rope_dims, 2, dtype=torch.float)
    theta = 1.0 / torch.pow(base, even / rope_dims)
    pos = torch.arange(seq_len, dtype=torch.float)
    angle = (pos[:, None] * theta[None, :]).repeat_interleave(2, dim=-1)
    cos, sin = torch.cos(angle), torch.sin(angle)
    zeta = (even + 0.4 * rope_dims) / (1.4 * rope_dims)
    center = (seq_len - 1 + 1) // 2
    power = (pos - center) / xpos_scale_base
    zq = (zeta ** power[:, None]).repeat_interleave(2, dim=-1)
    zk = (zeta ** (-power[:, None])).repeat_interleave(2, dim=-1)
    ones = torch.ones(seq_len, head_dim - rope_dims)
    zeros = torch.zeros(seq_len, head_dim - rope_dims)
    return RopeTables(
        cos_q=torch.cat([cos * zq, ones], dim=-1),
        sin_q=torch.cat([sin * zq, zeros], dim=-1),
        cos_k=torch.cat([cos * zk, ones], dim=-1),
        sin_k=torch.cat([sin * zk, zeros], dim=-1),
    )


def rotate_pairs(x: Tensor) -> Tensor:
    """``rot(x)`` on interleaved pairs over the whole last dim: ``(a, b) -> (-b, a)``."""
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack([-b, a], dim=-1).flatten(-2)


def pair_rotation(n: int = 32) -> Tensor:
    """``(n, n)`` matrix with ``x @ R == rotate_pairs(x)``, the ``rotary_embedding_llama`` transformation matrix."""
    r = torch.zeros(n, n)
    r[torch.arange(1, n, 2), torch.arange(0, n, 2)] = -1.0
    r[torch.arange(0, n, 2), torch.arange(1, n, 2)] = 1.0
    return r


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    return x * cos + rotate_pairs(x) * sin
