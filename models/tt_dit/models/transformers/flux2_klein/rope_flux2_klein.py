# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX.2 Klein RoPE tables for the device blocks (host torch; diffusers is never imported here).

Re-implements ``Flux2PosEmbed.forward`` (diffusers ``transformer_flux2.py:971-998``) with ``get_1d_rotary_pos_embed``
(``embeddings.py:885-949``) per axis: four position axes of 32 channels each (``axes_dims_rope``), ``theta = 2000``,
positions cast to float then promoted to float64 by the float64 inverse frequencies ``theta ** (-arange(0, 32, 2) /
32)``, ``cos`` / ``sin`` of the outer product, ``repeat_interleave(2)`` so channels ``2p, 2p + 1`` share pair ``p``'s
angle (``embeddings.py:938-939``), and the cast to float32 afterwards; the four axes are concatenated axis-major to
the 128 = ``head_dim`` channels. The BFL ``rope`` (``model.py:818-833``) rotates the same pairs. The fused QK-norm +
RoPE op (``normalization.py`` ``DistributedRMSNorm`` with ``get_rot_transformation_mat``) rotates ``out[2i] = x[2i] cos -
x[2i + 1] sin``, ``out[2i + 1] = x[2i + 1] cos + x[2i] sin``, which is ``apply_rotary_emb`` with
``use_real_unbind_dim=-1`` (``embeddings.py:986-997``).

Tables are fp32 ``[1, 1, padded, 128]`` with pad rows ``cos 1 / sin 0`` (a pad row's RoPE is the identity; its key is
masked by the ring SDPA anyway). Ids come from ``pipelines/flux2_klein/packing.py``: ``latent_ids`` ``(0, h, w, 0)``,
``reference_ids`` ``(10 + 10 i, h, w, 0)``, ``text_ids`` ``(0, 0, 0, l)``, all int64.

The single blocks run on ONE sequence per device, the per-device concatenation ``[prompt_d | spatial_d]`` of the two
SP shards (``transformer_flux2_klein.Flux2KleinTransformer.forward``); :func:`single_stream_rope` builds the matching
tables on the host so that, uploaded SP-sharded, device ``d`` holds exactly the rows of its concatenation.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import ttnn

from ....utils import tensor


def rope_cos_sin(
    ids: torch.Tensor, padded: int, *, theta: int, axes_dim: Sequence[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(cos, sin)`` fp32 ``[1, 1, padded, sum(axes_dim)]`` for ``ids`` ``[L, len(axes_dim)]`` (int or float),
    ``L <= padded``; rows ``[L, padded)`` carry ``cos 1 / sin 0``."""
    if ids.dim() != 2 or ids.shape[-1] != len(axes_dim):
        raise ValueError(f"ids must be [L, {len(axes_dim)}], got {tuple(ids.shape)}")
    length = ids.shape[0]
    if padded < length:
        raise ValueError(f"cannot pad {length} rows down to {padded}")
    rope_dim = sum(axes_dim)
    # transformer_flux2.py:985: pos = ids.float(); embeddings.py:928-932: float64 frequencies promote the outer product.
    pos = ids.float()
    cos_parts, sin_parts = [], []
    for axis, dim in enumerate(axes_dim):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
        angle = torch.outer(pos[:, axis], freqs)
        # embeddings.py:938-939: cos / sin in float64, interleaved pairs, then float32.
        cos_parts.append(angle.cos().repeat_interleave(2, dim=-1).float())
        sin_parts.append(angle.sin().repeat_interleave(2, dim=-1).float())
    cos = torch.cat(cos_parts, dim=-1)
    sin = torch.cat(sin_parts, dim=-1)
    pad = padded - length
    cos = torch.nn.functional.pad(cos, (0, 0, 0, pad), value=1.0)
    sin = torch.nn.functional.pad(sin, (0, 0, 0, pad), value=0.0)
    return cos.reshape(1, 1, padded, rope_dim), sin.reshape(1, 1, padded, rope_dim)


def interleave_sp_shards(prompt: torch.Tensor, spatial: torch.Tensor, *, sp_factor: int) -> torch.Tensor:
    """Rows of ``prompt`` ``[1, 1, P, C]`` and ``spatial`` ``[1, 1, L, C]`` re-ordered as the SP-sharded upload of the
    per-device concatenation ``[prompt_d | spatial_d]``: ``[1, 1, P + L, C]`` whose shard ``d`` (rows ``[d (P + L) /
    SP, (d + 1) (P + L) / SP)``) is ``cat(prompt rows [d P / SP, (d + 1) P / SP), spatial rows [d L / SP, (d + 1) L /
    SP))``. In the ring-gathered order ``[p_0 s_0 ... p_{SP-1} s_{SP-1}]`` the spatial tail (the pad rows) stays the
    global tail, so ``logical_n = P + L_real`` masks exactly the pad rows."""
    if (
        prompt.dim() != 4
        or spatial.dim() != 4
        or prompt.shape[:2] != spatial.shape[:2]
        or prompt.shape[3] != spatial.shape[3]
    ):
        raise ValueError(
            f"tables must be [1, 1, rows, C] with one C, got {tuple(prompt.shape)}, {tuple(spatial.shape)}"
        )
    if sp_factor < 1:
        raise ValueError(f"sp_factor must be positive, got {sp_factor}")
    rows_p, rows_s = prompt.shape[2], spatial.shape[2]
    if rows_p % sp_factor or rows_s % sp_factor:
        raise ValueError(f"{rows_p} prompt rows and {rows_s} spatial rows must both shard over SP {sp_factor}")
    p = prompt.split(rows_p // sp_factor, dim=2)
    s = spatial.split(rows_s // sp_factor, dim=2)
    return torch.cat([t for d in range(sp_factor) for t in (p[d], s[d])], dim=2)


def single_stream_rope(
    prompt: tuple[torch.Tensor, torch.Tensor], spatial: tuple[torch.Tensor, torch.Tensor], *, sp_factor: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(cos, sin)`` of the single blocks' one sequence from the prompt's and the spatial stream's
    :func:`rope_cos_sin` pairs (:func:`interleave_sp_shards` on each)."""
    cos_p, sin_p = prompt
    cos_s, sin_s = spatial
    return (
        interleave_sp_shards(cos_p, cos_s, sp_factor=sp_factor),
        interleave_sp_shards(sin_p, sin_s, sp_factor=sp_factor),
    )


def rope_to_device(
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    device: ttnn.MeshDevice,
    sp_axis: int,
    sharded: bool,
    dtype: ttnn.DataType = ttnn.float32,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Upload a :func:`rope_cos_sin` pair in the blocks' layout: rows fractured over ``sp_axis`` like the activation
    when ``sharded`` (the spatial stream, and the prompt under ``shard_prompt``), replicated otherwise; replicated
    over TP either way. fp32 by default (no non-reference rounding where it is free)."""
    mesh_axes = [None, None, sp_axis if sharded else None, None]
    upload = lambda t: tensor.from_torch(t, device=device, dtype=dtype, mesh_axes=mesh_axes)  # noqa: E731
    return upload(cos), upload(sin)
