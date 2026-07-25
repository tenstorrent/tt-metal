# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pin HF DiffusionGemma's ACTUAL sliding-layer denoise visibility (#51080).

The DG reference module used to document the rule as "attend only keys with
``abs(q_idx - kv_idx) <= sliding_window``" and implemented that staircase. HF does no such
thing. These tests pin what HF really does, so the per-layer-span work is validated against
the real reference and a transformers upgrade that changes it fails loudly here instead of
silently shifting committed tokens.

Ground truth, established by reading
``transformers/models/diffusion_gemma/modeling_diffusion_gemma.py`` and ``cache_utils.py``:

1. A sliding layer's cache RETAINS only ``sliding_window - 1`` committed tokens
   (``DynamicSlidingWindowLayer.update`` keeps ``full_key_states[:, :, -sliding_window + 1:, :]``).
2. For the ordinary unpadded DynamicCache path, ``create_diffusion_decoder_attention_mask``
   returns ``None`` for BOTH masks — so there is no sliding mask at all and the window is
   purely a cache-truncation effect.
3. When a padding mask IS materialized, the sliding mask is expanded from a 1-D per-key
   vector, so it has NO query-index dependence — every canvas row sees the same key set.

=> HF sliding-layer denoise visibility = the ``sliding_window - 1`` most recent committed
   tokens, ALL-ATTEND, plus the full canvas. No staircase.

CPU-only; no device and no checkpoint required.
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")


W = 8  # tiny stand-in for the real sliding_window=1024; the arithmetic is what matters
CANVAS = 4


def _kv(seq_len: int, *, offset: int = 0):
    """K/V shaped [batch, heads, seq, head_dim] whose values encode absolute position."""
    pos = torch.arange(offset, offset + seq_len, dtype=torch.float32)
    return pos.view(1, 1, seq_len, 1).clone(), pos.view(1, 1, seq_len, 1).clone()


def test_sliding_layer_cache_retains_exactly_window_minus_one():
    """Fact 1: retention is sliding_window - 1, NOT sliding_window."""
    from transformers.cache_utils import DynamicSlidingWindowLayer

    layer = DynamicSlidingWindowLayer(sliding_window=W)
    n = 5 * W
    k, v = _kv(n)
    out_k, _out_v = layer.update(k, v)

    # The op returns the full concatenation for this call, but what it RETAINS is truncated.
    assert layer.keys.shape[2] == W - 1, f"expected {W - 1} retained rows, got {layer.keys.shape[2]}"
    retained = layer.keys.flatten().tolist()
    assert retained == list(range(n - (W - 1), n)), "retained rows must be the most recent W-1 positions"
    assert out_k.shape[2] >= layer.keys.shape[2]


def test_unpadded_dynamic_cache_produces_no_sliding_mask_at_all():
    """Fact 2: the common DG path returns None for both masks -> no staircase exists."""
    from transformers.cache_utils import DynamicCache
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaDecoderModel

    config = _text_config()
    past = DynamicCache(config=config)
    prompt_len = 3 * W
    _seed_cache(past, config, prompt_len)

    inputs_embeds = torch.zeros(1, CANVAS, config.hidden_size)

    # decoder_attention_mask=None -> shortcut return
    masks = DiffusionGemmaDecoderModel.create_diffusion_decoder_attention_mask(
        config=config, inputs_embeds=inputs_embeds, past_key_values=past, decoder_attention_mask=None
    )
    assert masks == {"full_attention": None, "sliding_attention": None}

    # an all-ones (unpadded) mask takes the same shortcut
    all_ones = torch.ones(1, prompt_len + CANVAS, dtype=torch.long)
    masks = DiffusionGemmaDecoderModel.create_diffusion_decoder_attention_mask(
        config=config, inputs_embeds=inputs_embeds, past_key_values=past, decoder_attention_mask=all_ones
    )
    assert masks == {"full_attention": None, "sliding_attention": None}, (
        "an unpadded DynamicCache must produce no materialized mask; the sliding window is "
        "a cache-truncation effect, not a mask"
    )


def test_materialized_sliding_mask_has_no_query_dependence_and_spans_window_minus_one():
    """Fact 3: when materialized, the sliding mask is query-INDEPENDENT (no staircase)."""
    from transformers.cache_utils import DynamicCache
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaDecoderModel

    config = _text_config()
    past = DynamicCache(config=config)
    prompt_len = 3 * W
    _seed_cache(past, config, prompt_len)

    inputs_embeds = torch.zeros(1, CANVAS, config.hidden_size)
    # One left-pad position forces materialization (mask.all() is False).
    padded = torch.ones(1, prompt_len + CANVAS, dtype=torch.long)
    padded[0, 0] = 0

    masks = DiffusionGemmaDecoderModel.create_diffusion_decoder_attention_mask(
        config=config, inputs_embeds=inputs_embeds, past_key_values=past, decoder_attention_mask=padded
    )
    sliding = masks["sliding_attention"]
    assert sliding is not None

    # NO staircase: every canvas (query) row must see an identical key set.
    rows = sliding.shape[2]
    assert rows == CANVAS
    for i in range(1, rows):
        assert torch.equal(
            sliding[:, :, 0, :], sliding[:, :, i, :]
        ), "sliding mask varies by query row -- that would be a |q-k| staircase, which HF does not do"

    # Span: (W-1) committed columns + the canvas, all canvas columns attended.
    assert sliding.shape[-1] == (W - 1) + CANVAS
    assert bool(sliding[..., -CANVAS:].all()), "canvas columns must be fully attended"


def test_tt_window_span_formula_matches_hf_key_set():
    """The span/offset arithmetic the TT per-layer read will use reproduces HF's key set.

    TT reads a TILE-ALIGNED ``span`` rows at ``lo = max(0, P - span)`` and masks the
    uncommitted / out-of-window columns. HF attends absolute positions ``[P-(W-1), P)``.
    A tile-aligned span cannot be ``W-1`` (1023 is not a multiple of 32), which is exactly
    why the design reads one extra row and masks it out.
    """
    span = W  # stands in for the tile-aligned 1024 vs HF's 1023

    for P in (1, W - 1, W, W + 1, 2 * W, 5 * W):
        hf_visible = set(range(max(0, P - (W - 1)), P))

        lo = max(0, P - span)
        tt_visible = {
            lo + r
            for r in range(span)
            # committed-prefix predicate AND the HF cache-retention predicate
            if (lo + r) < P and (lo + r) >= P - (W - 1)
        }
        assert tt_visible == hf_visible, f"P={P}: TT {sorted(tt_visible)} != HF {sorted(hf_visible)}"


# --------------------------------------------------------------------------------------------
# helpers


def _text_config():
    from transformers.models.diffusion_gemma.configuration_diffusion_gemma import DiffusionGemmaTextConfig

    return DiffusionGemmaTextConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        sliding_window=W,
        canvas_length=CANVAS,
        layer_types=["sliding_attention", "full_attention"],
    )


def _seed_cache(past, config, prompt_len: int):
    """Push ``prompt_len`` committed tokens through every layer so get_seq_length() == prompt_len."""
    k, v = _kv(prompt_len)
    k = k.expand(1, config.num_key_value_heads, prompt_len, config.head_dim).contiguous()
    v = v.expand(1, config.num_key_value_heads, prompt_len, config.head_dim).contiguous()
    for layer_idx in range(config.num_hidden_layers):
        past.update(k.clone(), v.clone(), layer_idx)
