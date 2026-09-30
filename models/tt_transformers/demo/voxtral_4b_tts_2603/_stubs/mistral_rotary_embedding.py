# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `mistral_rotary_embedding` (`model.rotary_emb`).

`(cos, sin)` for a batch of positions, where `cos = cat(freqs, freqs).cos()` and
`freqs = position (x) inv_freq` -- `rotate_half` convention, head_dim 128, `rope_theta` **1e6**.

Implemented as a lookup rather than an on-device outer product. `cos`/`sin` for positions
0..`_MAX_POSITIONS` are materialised on the host **by calling this very module**, so the table is
the reference's own output rather than a re-derivation that has to guess whether `rope_theta` lives
at `config.rope_theta` (transformers 4.x) or inside `config.rope_parameters` (5.x) -- getting that
wrong silently runs at the default 1e4 instead of 1e6. `position_ids` then gathers rows with
`ttnn.embedding`, which keeps arbitrary (non-contiguous, decode-style) positions working while the
forward makes no torch call at all: the runtime native probe graduates only at zero torch ops.

The `x` argument is positional-first and the reference only reads its dtype/device off it, so it is
accepted and unused here too.
"""

from __future__ import annotations

import torch

import ttnn


_MAX_POSITIONS = 8192


def _from_torch(t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    t = t.to(torch.bfloat16) if dtype == ttnn.bfloat16 else t.to(torch.float32)
    if device.__class__.__name__ == "MeshDevice":
        return ttnn.from_torch(
            t, dtype=dtype, layout=layout, device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


def build(device, torch_module):
    rotary = torch_module

    positions = torch.arange(_MAX_POSITIONS, dtype=torch.long).unsqueeze(0)
    probe = torch.zeros(1, _MAX_POSITIONS, 1, dtype=torch.float32)
    with torch.no_grad():
        cos, sin = rotary(probe, positions)
    head_dim = int(cos.shape[-1])

    # `ttnn.embedding` requires a bfloat16 table (`embedding_device_operation.cpp:36`), so the
    # GATHER path pays a bfloat16 round-trip it cannot avoid.
    cos_table = _from_torch(cos[0].contiguous(), device, layout=ttnn.ROW_MAJOR_LAYOUT)
    sin_table = _from_torch(sin[0].contiguous(), device, layout=ttnn.ROW_MAJOR_LAYOUT)
    # The CONTIGUOUS path is a plain slice with no such constraint, so it keeps float32 --
    # `_from_torch`'s default is bfloat16 and taking it here quietly made rope the only
    # bfloat16 term in an otherwise float32 residual stream. That is ~0.4% (one bfloat16 ulp)
    # per rotated q/k, and it compounds: over a 26-layer prefill the last hidden state lands at
    # PCC 0.9971 instead of 0.9999, which a 21-level acoustic quantiser turns into wrong codes.
    # `mistral_model.py` builds this same pair float32 for the same reason.
    cos_tiled = _from_torch(cos.reshape(1, _MAX_POSITIONS, head_dim).contiguous(), device, dtype=ttnn.float32)
    sin_tiled = _from_torch(sin.reshape(1, _MAX_POSITIONS, head_dim).contiguous(), device, dtype=ttnn.float32)
    # ROW_MAJOR float32, for the SINGLE-POSITION path below. A decode step's positions are one
    # scalar shared by every user, so it needs a row of the table, not a gather -- and a row can be
    # taken with a plain slice, which carries no dtype constraint. TILE layout would: its -2 axis
    # is tiled 32 rows at a time, so slicing row 33 out of the tiled table is not a free view.
    cos_rows = _from_torch(
        cos.reshape(1, _MAX_POSITIONS, head_dim).contiguous(), device,
        dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    sin_rows = _from_torch(
        sin.reshape(1, _MAX_POSITIONS, head_dim).contiguous(), device,
        dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    def mistral_rotary_embedding(x, position_ids=None, position=None, **kwargs):
        if position is not None:
            # ONE position, every user. `ttnn.embedding` needs a bfloat16 table, and taking the
            # gather here for a single row made RoPE the only bfloat16 term in a float32 residual
            # stream -- the same ~0.4% per rotated q/k that the contiguous path was moved to
            # float32 to escape, except a decode step pays it at EVERY layer with no prefill to
            # dilute it. Measured: the cached decode step sat at PCC ~0.978 against the reference
            # while the prefill through the same weights was at 0.99994.
            p = int(position)
            return (
                ttnn.slice(cos_rows, [0, p, 0], [1, p + 1, head_dim]),
                ttnn.slice(sin_rows, [0, p, 0], [1, p + 1, head_dim]),
            )
        if position_ids is None:
            # The reference defaults to contiguous positions, which are the table's first rows.
            seq = int(x.shape[-2])
            return (
                ttnn.slice(cos_tiled, [0, 0, 0], [1, seq, head_dim]),
                ttnn.slice(sin_tiled, [0, 0, 0], [1, seq, head_dim]),
            )
        return (
            ttnn.embedding(position_ids, cos_table, layout=ttnn.TILE_LAYOUT),
            ttnn.embedding(position_ids, sin_table, layout=ttnn.TILE_LAYOUT),
        )

    return mistral_rotary_embedding
