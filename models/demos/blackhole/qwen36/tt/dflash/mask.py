# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Attention visibility for the DFlash drafter.

The drafter's attention is an unusual shape: queries are only the ``block_len`` block
positions, while keys/values are ``concat(context, block)``. So query row ``i`` sits at
absolute position ``ctx_len + i`` and the mask is rectangular, not square.

This module reproduces ``_attention_mask`` in ``reference/dflash/dflash.py:159`` together
with the guard at ``:395`` that decides whether a mask is built at all. It is kept separate
from the attention op, and pure-torch, because the per-layer asymmetry here is the single
easiest thing to get wrong in this port and it is worth being able to test with no device:

* ``sliding_attention`` layers are **causal and windowed** -- ``is_causal`` is derived as
  ``layer_type == "sliding_attention"``.
* the ``full_attention`` layer is **non-causal and unwindowed**, so the reference builds no
  mask at all and the layer attends bidirectionally over context + block. For
  Qwen3.6-27B-DFlash that single layer is where all of the "block diffusion"
  bidirectionality lives.

Getting this backwards is not a subtle regression, but neither is it loud: the
Muse-Glimmer port recorded (work_log F3b) an *unwindowed* reimplementation scoring 0.99997
against a golden that had itself accidentally lost its window, while the correct
implementation scored 0.9294 against the same golden. A mask bug grades as a port bug. The
tests in ``tests/dflash/test_mask.py`` therefore assert the semantics directly rather than
inferring them from PCC.

qwen36's own attention never builds a mask -- it relies on SDPA's ``is_causal`` and the
paged decode variants -- so there is no in-repo convention to match. Note that SDPA's
built-in ``is_causal`` cannot be used for the sliding layers here: it assumes queries and
keys share an origin, whereas our queries start at ``ctx_len``.
"""

from __future__ import annotations

import torch

from models.demos.blackhole.qwen36.tt.dflash.config import DFlashDrafterConfig

#: Additive fill for masked positions. Well below any attainable score, and exactly
#: representable in bfloat16 (unlike ``torch.finfo(bfloat16).min``, which invites inf-inf
#: NaNs if a row ever ends up fully masked). No row here is ever fully masked -- a query
#: always sees at least itself -- but the margin is free.
MASK_NEG = -1e9


def visibility(
    cfg: DFlashDrafterConfig,
    layer_idx: int,
    *,
    ctx_len: int,
    block_len: int,
) -> torch.Tensor | None:
    """Boolean ``[block_len, ctx_len + block_len]`` mask; ``True`` where a key is visible.

    Returns ``None`` when the layer needs no mask at all (the ``full_attention`` case),
    matching the reference, where ``attention_mask`` stays ``None`` and SDPA attends over
    everything.
    """
    is_causal = cfg.is_causal(layer_idx)
    window = cfg.window_for(layer_idx)
    if not is_causal and window is None:
        return None

    key_len = ctx_len + block_len
    # Queries are the LAST block_len positions of the key sequence.
    query_position = key_len - block_len + torch.arange(block_len)[:, None]
    key_position = torch.arange(key_len)[None, :]

    visible = torch.ones((block_len, key_len), dtype=torch.bool)
    if is_causal:
        visible &= key_position <= query_position
    if window is not None:
        visible &= query_position - key_position < window
        if not is_causal:
            visible &= key_position - query_position < window
    return visible


def additive(
    cfg: DFlashDrafterConfig,
    layer_idx: int,
    *,
    ctx_len: int,
    block_len: int,
    dtype: torch.dtype = torch.float32,
    neg: float = MASK_NEG,
) -> torch.Tensor | None:
    """:func:`visibility` as an additive ``[1, 1, block_len, ctx_len + block_len]`` mask.

    ``0`` where visible, ``neg`` where not. ``None`` passes through unchanged.
    """
    visible = visibility(cfg, layer_idx, ctx_len=ctx_len, block_len=block_len)
    if visible is None:
        return None
    return torch.where(visible, 0.0, neg).to(dtype)[None, None]


def context_keep_from(
    cfg: DFlashDrafterConfig,
    layer_idx: int,
    *,
    ctx_len: int,
) -> int:
    """First context index any query on this layer can see.

    A windowed layer's earliest query is at absolute position ``ctx_len`` and sees keys
    ``> ctx_len - window``, so every context key below ``ctx_len - window + 1`` is
    invisible to *all* queries and can be dropped before the matmul. That turns the
    sliding layers' score matrix from ``block_len x (ctx_len + block_len)`` into
    ``block_len x (window + block_len)`` -- 16x2064 at window 2048 -- which is why the
    bring-up path can use a dense mask instead of a windowed SDPA kernel.

    Returns 0 for layers with no window (the ``full_attention`` layer must see the whole
    context and cannot be sliced).
    """
    window = cfg.window_for(layer_idx)
    if window is None:
        return 0
    return max(0, ctx_len - window + 1)


def to_device(
    cfg: DFlashDrafterConfig,
    layer_idx: int,
    mesh_device,
    *,
    ctx_len: int,
    block_len: int,
    dtype=None,
    neg: float = MASK_NEG,
):
    """Build this layer's additive mask as a replicated device tensor, or ``None``.

    Call at construction, once per (layer kind, ctx length) -- never inside a forward. The
    drafter's inference path must not touch host, and a mask is a pure function of shape, so
    there is no reason for it to.

    Shape is ``[1, 1, block_len, ctx_len + block_len]``, which ``ttnn.transformer
    .scaled_dot_product_attention`` broadcasts over batch and head.
    """
    import ttnn

    additive_mask = additive(cfg, layer_idx, ctx_len=ctx_len, block_len=block_len, neg=neg)
    if additive_mask is None:
        return None
    return ttnn.from_torch(
        additive_mask.to(torch.bfloat16),
        dtype=dtype or ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
