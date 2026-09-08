# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mask semantics for the DFlash drafter. No device, no checkpoint, no network.

The centrepiece is :func:`test_matches_reference`, a differential test against the
vendored upstream ``_attention_mask`` -- the actual oracle. The hand-written assertions
after it are not redundant: they pin the *intent* in a form that survives the reference
being upgraded, and they are what makes a failure legible ("the full layer went causal")
rather than just "tensors differ".

Why this file is the cheapest and most valuable gate in the suite: the Muse-Glimmer DFlash
port recorded (work_log F3b) an *unwindowed* reimplementation scoring 0.99997 against a
golden that had itself silently lost its window, while the correct implementation scored
0.9294 against that same golden. A mask bug presents as a port bug, and PCC cannot tell
you which you have. So assert the semantics directly.
"""

from __future__ import annotations

import pytest
import torch

from models.demos.blackhole.qwen36.reference.dflash.dflash import _attention_mask
from models.demos.blackhole.qwen36.tt.dflash import mask as dflash_mask

# (ctx_len, block_len). block_len 16 is the real block_size; ctx 0 is the first-block case
# (no context yet), 2047/2048/2049 straddle the sliding window, and 4096 is comfortably
# past it -- the only regime where the window's lower bound actually bites.
SHAPES = [
    pytest.param(0, 16, id="ctx0"),
    pytest.param(64, 16, id="ctx64"),
    pytest.param(2047, 16, id="ctx2047"),
    pytest.param(2048, 16, id="ctx2048"),
    pytest.param(2049, 16, id="ctx2049"),
    pytest.param(4096, 16, id="ctx4096"),
]

SLIDING_LAYERS = (0, 1, 2, 3)
FULL_LAYER = 4


def _reference_visibility(cfg, layer_idx: int, ctx_len: int, block_len: int):
    """What the reference actually produces, including its call guard.

    ``Qwen3DFlashAttention.forward`` only calls ``_attention_mask`` when
    ``attention_mask is None and (self.is_causal or self.sliding_window is not None)``
    (``reference/dflash/dflash.py:395``); otherwise the mask stays ``None`` and SDPA
    attends over everything. Reproducing the guard is the point -- the ``None`` case *is*
    the full-attention layer's semantics.
    """
    is_causal = cfg.is_causal(layer_idx)
    window = cfg.window_for(layer_idx)
    if not (is_causal or window is not None):
        return None
    query = torch.zeros(1, 1, block_len, 1)
    key = torch.zeros(1, 1, ctx_len + block_len, 1)
    return _attention_mask(query, key, is_causal=is_causal, sliding_window=window)[0, 0]


@pytest.mark.parametrize("ctx_len, block_len", SHAPES)
def test_matches_reference(drafter_cfg, ctx_len, block_len):
    """Every layer's mask is bit-identical to the upstream reference's."""
    for layer_idx in range(drafter_cfg.num_hidden_layers):
        expected = _reference_visibility(drafter_cfg, layer_idx, ctx_len, block_len)
        actual = dflash_mask.visibility(drafter_cfg, layer_idx, ctx_len=ctx_len, block_len=block_len)

        if expected is None:
            assert actual is None, f"layer {layer_idx}: reference builds no mask but we built one"
            continue
        assert actual is not None, f"layer {layer_idx}: reference builds a mask but we returned None"
        assert actual.shape == expected.shape, f"layer {layer_idx}: {actual.shape} != {expected.shape}"
        assert torch.equal(actual, expected), f"layer {layer_idx}: mask differs from reference"


def test_layer_types_are_as_published(drafter_cfg):
    """The 4-sliding + 1-full split is what the rest of this file assumes."""
    assert drafter_cfg.num_hidden_layers == 5
    assert all(drafter_cfg.is_sliding(i) for i in SLIDING_LAYERS)
    assert not drafter_cfg.is_sliding(FULL_LAYER)
    assert drafter_cfg.sliding_window == 2048


@pytest.mark.parametrize("layer_idx", SLIDING_LAYERS)
def test_sliding_layers_are_causal(drafter_cfg, layer_idx):
    """A sliding layer's query cannot see a later block slot.

    This is the assertion that separates this checkpoint from Muse-Glimmer's, whose port
    runs every layer bidirectionally-windowed. Ours derives
    ``is_causal = (layer_type == "sliding_attention")``, so the sliding layers are causal.
    """
    ctx_len, block_len = 64, 16
    visible = dflash_mask.visibility(drafter_cfg, layer_idx, ctx_len=ctx_len, block_len=block_len)

    # Query row 6 is absolute position ctx_len + 6; block slot 9 is key ctx_len + 9.
    assert not visible[6, ctx_len + 9], "sliding layer let a query see a later block slot"
    assert visible[6, ctx_len + 6], "query cannot see itself"
    assert visible[6, ctx_len + 3], "query cannot see an earlier block slot"
    # Whole-mask form of the same property.
    assert not visible[:, ctx_len:].triu(diagonal=1).any(), "sliding layer is not causal within the block"


def test_full_layer_is_bidirectional(drafter_cfg):
    """The full-attention layer has no mask at all -- this is where block diffusion lives.

    ``is_causal=False`` *and* ``sliding_window=None`` means the reference never builds a
    mask, so every masked slot sees every other slot and the whole context. If this ever
    starts returning a mask, the drafter has silently become autoregressive and acceptance
    will fall without any PCC gate necessarily catching it.
    """
    assert dflash_mask.visibility(drafter_cfg, FULL_LAYER, ctx_len=64, block_len=16) is None
    assert dflash_mask.additive(drafter_cfg, FULL_LAYER, ctx_len=64, block_len=16) is None
    assert not drafter_cfg.is_causal(FULL_LAYER)
    assert drafter_cfg.window_for(FULL_LAYER) is None
    # And it must not be sliceable -- it needs the entire context.
    assert dflash_mask.context_keep_from(drafter_cfg, FULL_LAYER, ctx_len=4096) == 0


@pytest.mark.parametrize("layer_idx", SLIDING_LAYERS)
def test_sliding_window_boundary_is_exclusive(drafter_cfg, layer_idx):
    """``query_pos - key_pos < window``: distance ``window - 1`` is in, ``window`` is out."""
    window = drafter_cfg.sliding_window
    ctx_len, block_len = 4096, 16
    visible = dflash_mask.visibility(drafter_cfg, layer_idx, ctx_len=ctx_len, block_len=block_len)

    q_abs = ctx_len  # query row 0
    assert visible[0, q_abs - (window - 1)], f"key at distance {window - 1} should be visible"
    assert not visible[0, q_abs - window], f"key at distance {window} should be masked"
    # Exactly `window` keys are visible to each query once the window is saturated.
    assert visible.sum(dim=-1).unique().tolist() == [window]


@pytest.mark.parametrize("ctx_len, block_len", SHAPES)
def test_no_row_is_fully_masked(drafter_cfg, ctx_len, block_len):
    """Every query sees at least itself, so softmax can never see an all-``-inf`` row.

    :data:`~models.demos.blackhole.qwen36.tt.dflash.mask.MASK_NEG` relies on this.
    """
    for layer_idx in range(drafter_cfg.num_hidden_layers):
        visible = dflash_mask.visibility(drafter_cfg, layer_idx, ctx_len=ctx_len, block_len=block_len)
        if visible is None:
            continue
        assert visible.any(dim=-1).all(), f"layer {layer_idx}: a query row is fully masked"


@pytest.mark.parametrize("ctx_len, block_len", SHAPES)
@pytest.mark.parametrize("layer_idx", SLIDING_LAYERS)
def test_context_slicing_drops_only_invisible_keys(drafter_cfg, layer_idx, ctx_len, block_len):
    """Slicing the context to ``context_keep_from`` is lossless.

    This is what lets the sliding layers use a dense 16x2064 mask instead of a windowed
    SDPA kernel: everything dropped was already masked for every query.
    """
    keep_from = dflash_mask.context_keep_from(drafter_cfg, layer_idx, ctx_len=ctx_len)
    visible = dflash_mask.visibility(drafter_cfg, layer_idx, ctx_len=ctx_len, block_len=block_len)

    assert not visible[:, :keep_from].any(), "sliced-away context keys were visible to some query"

    # And the mask built for the sliced context equals the surviving columns of the full one.
    sliced = dflash_mask.visibility(drafter_cfg, layer_idx, ctx_len=ctx_len - keep_from, block_len=block_len)
    assert torch.equal(sliced, visible[:, keep_from:])

    kept = ctx_len - keep_from + block_len
    assert kept <= drafter_cfg.sliding_window + block_len, "slice did not bound the score matrix"


@pytest.mark.parametrize("layer_idx", SLIDING_LAYERS)
def test_additive_mask_encoding(drafter_cfg, layer_idx):
    """``additive`` is ``0`` where visible and ``MASK_NEG`` where not, shaped for SDPA."""
    ctx_len, block_len = 64, 16
    visible = dflash_mask.visibility(drafter_cfg, layer_idx, ctx_len=ctx_len, block_len=block_len)
    additive = dflash_mask.additive(drafter_cfg, layer_idx, ctx_len=ctx_len, block_len=block_len, dtype=torch.bfloat16)

    assert additive.shape == (1, 1, block_len, ctx_len + block_len)
    assert additive.dtype == torch.bfloat16
    assert (additive[0, 0][visible] == 0).all()
    assert (additive[0, 0][~visible] == dflash_mask.MASK_NEG).all()
    # MASK_NEG must survive the cast to the compute dtype without becoming -inf.
    assert torch.isfinite(additive).all()
