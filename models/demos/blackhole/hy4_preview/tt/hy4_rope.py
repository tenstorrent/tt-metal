# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""RoPE tables for Hy4 (CPU, torch). Rotation is ``x * C + (x @ P) * S`` over a 64-wide vector with full-width tables, for either pairing:

* ``interleave=True``  adjacent pairs (x[2i], x[2i+1])      (DeepSeek-V3 HF convention; oracle default for the MLA rope)
* ``interleave=False`` halves      (x[i], x[i+32])           (neox / rotate_half; the oracle default for the indexer rope)

Which one the checkpoint uses is an open question of the ref-oracle (variant A/B); the device code takes the flag, nothing else changes.
"""
import torch


def rope_tables(positions, dim=64, theta=1e7, interleave=True):
    """positions [n] (int) -> cos, sin [n, dim] (fp32), the pair / half repeat already applied."""
    pos = torch.as_tensor(positions, dtype=torch.float32)
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    ang = torch.outer(pos, inv)
    c, s = ang.cos(), ang.sin()
    if interleave:
        return c.repeat_interleave(2, dim=-1), s.repeat_interleave(2, dim=-1)
    return torch.cat([c, c], -1), torch.cat([s, s], -1)


def rotation_matrix(dim=64, interleave=True):
    """P [dim, dim] with (x @ P) = the 'rotated' vector: interleave: (x@P)[2i] = -x[2i+1], (x@P)[2i+1] = x[2i]; half: (x@P)[i] = -x[i+h], (x@P)[i+h] = x[i]."""
    P = torch.zeros(dim, dim)
    if interleave:
        for i in range(dim // 2):
            P[2 * i + 1, 2 * i] = -1.0
            P[2 * i, 2 * i + 1] = 1.0
    else:
        h = dim // 2
        for i in range(h):
            P[i + h, i] = -1.0
            P[i, i + h] = 1.0
    return P


def apply_rope_host(x, cos, sin, interleave=True):
    """x [..., n, dim], cos/sin [n, dim] -> rotated (reference for tests and for host-side cache preparation)."""
    P = rotation_matrix(x.shape[-1], interleave)
    return x * cos + (x @ P) * sin
