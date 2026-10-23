# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only regression tests for `decoder_layer_kwargs`. No device, no weights, milliseconds.

Guards `fix(prefill): compare the KVPE the device actually caches`, which binds the reference layer
call by the layer's OWN signature. Getting it wrong fails silently: the cache kwarg lands in
**kwargs under the wrong name and no KV is ever captured.
"""

import torch

from models.demos.deepseek_v3_d_p.utils.transformer_helpers import decoder_layer_kwargs


class _OldStyleLayer(torch.nn.Module):
    """A vendored DeepSeek/Kimi/GLM-shaped layer: singular cache kwarg, rope computed internally."""

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=True):
        raise AssertionError("not called")


class _NewStyleLayer(torch.nn.Module):
    """A transformers-5.x layer: plural cache kwarg, and rope must be handed in."""

    def forward(
        self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None, position_embeddings=None
    ):
        raise AssertionError("not called")


def test_old_style_layer_kwargs_match_the_explicit_call_they_replaced():
    """`decoder_layer_kwargs` now binds the reference call in test_prefill_block for EVERY model, so
    on a vendored layer it must still reproduce the explicit call it replaced -- a silent drift here
    moves the reference those models' PCC rows are judged against.
    """
    cache = object()
    mask = torch.zeros(1, 1, 4, 4)
    pos = torch.arange(4).unsqueeze(0)
    kwargs = decoder_layer_kwargs(_OldStyleLayer(), None, torch.zeros(1, 4, 8), mask, pos, cache)

    assert set(kwargs) == {"attention_mask", "position_ids", "past_key_value", "use_cache"}
    assert kwargs["past_key_value"] is cache, "the vendored layers take the SINGULAR cache kwarg"
    assert kwargs["use_cache"] is True
    assert kwargs["attention_mask"] is mask and kwargs["position_ids"] is pos

    # The plural branch, which nothing else covers: a transformers-5.x layer must get the cache under
    # `past_key_values`, or the KV lands in **kwargs and is silently never captured.
    plural = decoder_layer_kwargs(
        _NewStyleLayer(), None, torch.zeros(1, 4, 8), mask, pos, cache, position_embeddings=(mask, mask)
    )
    assert "past_key_values" in plural and plural["past_key_values"] is cache


def test_an_old_style_layer_is_not_handed_position_embeddings():
    """A vendored layer computes rope itself and its model exposes no top-level ``rotary_emb``, so
    binding its kwargs must not reach ``reference_rope`` at all -- if it does, the assert there
    fires."""

    class _ModelWithoutRotary(torch.nn.Module):
        pass

    kwargs = decoder_layer_kwargs(
        _OldStyleLayer(),
        _ModelWithoutRotary(),
        torch.zeros(1, 4, 8),
        None,
        torch.arange(4).unsqueeze(0),
        None,
    )
    assert "position_embeddings" not in kwargs, "an old-style layer must not be handed position_embeddings"
