# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX 3 Image Klein RoPE tables for the device block (host torch; the reference package is never imported here).

Re-implements ``rope`` / ``rope_embed`` of ``reference/flux3_klein/dit.py:50-61``: four position axes of 32 channels
each (``AXES_DIM``), ``theta = 10000``, the angles ``pos * theta ** (-arange(0, 32, 2) / 32)`` in float64 and the
rotation entries cast to float32 afterwards. The block (``transformer_block_flux3_klein.py``, "rope" of its tensor
contract) consumes ``cos = emb[..., 0, 0]`` and ``sin = emb[..., 1, 0]`` of the reference's ``(1, 1, L, 64, 2, 2)``
matrices, each ``repeat_interleave(2, dim=-1)`` so channels ``2p, 2p + 1`` share pair ``p``'s angle, pairs axis-major,
as fp32 ``[1, 1, L_pad, 128]`` with pad rows ``cos 1 / sin 0`` (a pad row's RoPE is the identity; its key is masked by
the ring SDPA anyway). :func:`rope_cos_sin` is bit-equal to that conversion (``test_rope_flux3_klein.py``).

Ids come from ``pipelines/flux3_klein/packing.py``: ``image_ids`` ``(0, 0.5 row, 0.5 col, 0)`` fp32, ``reference_ids``
with ``l = -(1 + k)``, ``text_ids`` ``(0, 0, 0, pos)`` int64; int64 and fp32 ids give identical tables because every
value is exactly representable (the reference promotes the single-stage concatenation to fp32, ``dit.py:230``). One
table per non-empty mode at ``padded_length(L_m, SP)`` for the early stacks, one for the single-stage sequence from
the ``[image | image_cond | text]`` concatenation of the non-empty modes' ids at ``L_pad``.
"""

from __future__ import annotations

import torch

import ttnn

from ....utils import tensor

# dit.py:22-24
FLUX3_KLEIN_ROPE_THETA = 10000
FLUX3_KLEIN_ROPE_AXES_DIM = (32, 32, 32, 32)
FLUX3_KLEIN_ROPE_DIM = sum(FLUX3_KLEIN_ROPE_AXES_DIM)  # 128 = the attention head dim


def rope_cos_sin(ids: torch.Tensor, padded: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(cos, sin)`` fp32 ``[1, 1, padded, 128]`` for ``ids`` ``[L, 4]`` (int or float), ``L <= padded``; rows
    ``[L, padded)`` carry ``cos 1 / sin 0``."""
    if ids.dim() != 2 or ids.shape[-1] != len(FLUX3_KLEIN_ROPE_AXES_DIM):
        raise ValueError(f"ids must be [L, {len(FLUX3_KLEIN_ROPE_AXES_DIM)}], got {tuple(ids.shape)}")
    length = ids.shape[0]
    if padded < length:
        raise ValueError(f"cannot pad {length} rows down to {padded}")
    pos = ids.to(torch.float64)
    cos_parts, sin_parts = [], []
    for axis, dim in enumerate(FLUX3_KLEIN_ROPE_AXES_DIM):
        # dit.py:51-52, op for op: float64 angles (einsum of the positions with the inverse frequencies).
        scale = torch.arange(0, dim, 2, dtype=torch.float64) / dim
        angle = torch.einsum("...n,d->...nd", pos[:, axis], 1.0 / (FLUX3_KLEIN_ROPE_THETA**scale))
        cos_parts.append(torch.cos(angle))
        sin_parts.append(torch.sin(angle))
    # dit.py:53-54: the rotation entries are cast to float32 after the float64 trig; the pairs are axis-major.
    cos = torch.cat(cos_parts, dim=-1).float().repeat_interleave(2, dim=-1)
    sin = torch.cat(sin_parts, dim=-1).float().repeat_interleave(2, dim=-1)
    pad = padded - length
    cos = torch.nn.functional.pad(cos, (0, 0, 0, pad), value=1.0)
    sin = torch.nn.functional.pad(sin, (0, 0, 0, pad), value=0.0)
    return cos.reshape(1, 1, padded, FLUX3_KLEIN_ROPE_DIM), sin.reshape(1, 1, padded, FLUX3_KLEIN_ROPE_DIM)


def rope_to_device(
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    device: ttnn.MeshDevice,
    sp_axis: int,
    dtype: ttnn.DataType = ttnn.float32,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Upload a :func:`rope_cos_sin` pair in the block's layout: ``mesh_axes=[None, None, sp_axis, None]`` (rows
    fractured over SP like the activation, replicated over TP). fp32 by default (PLAN decision 10: no non-reference
    rounding where it is free)."""
    upload = lambda t: tensor.from_torch(
        t, device=device, dtype=dtype, mesh_axes=[None, None, sp_axis, None]
    )  # noqa: E731
    return upload(cos), upload(sin)
