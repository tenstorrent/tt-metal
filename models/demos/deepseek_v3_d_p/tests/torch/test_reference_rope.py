# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only regression tests for the fp32 reference rope. No device, no weights, milliseconds.

Fails on the code as it stood before `fix(mla): build the reference rope from the config, not the
model`. The bug is silent -- it produces a plausible wrong number rather than an error -- so the
first test also asserts the *pre-fix* behaviour is detectably wrong, otherwise it could pass for the
wrong reason.
"""

from copy import deepcopy

import pytest
import torch

from models.demos.deepseek_v3_d_p.utils.transformer_helpers import (
    layer_wants_position_embeddings,
    reference_position_embeddings,
    reference_rope,
)


def test_reference_rope_is_fp32_even_when_the_model_is_bf16():
    """Guards `fix(mla): build the reference rope from the config, not the model`.

    A reference instantiated in bf16 carries a bf16 ``inv_freq`` BUFFER, and ``.float()`` cannot
    recover bits already gone (0.75 where the true value is 0.7498942). The residue is a
    per-dimension frequency error, i.e. a phase error that grows LINEARLY with position -- harmless
    at short prompts, ruinous at long ones. Rebuilding from the config in fp32 is the only fix.
    """
    transformers = pytest.importorskip("transformers")
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    cfg = transformers.LlamaConfig(
        hidden_size=256, num_attention_heads=4, num_hidden_layers=1, rope_theta=10000.0, max_position_embeddings=65536
    )

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = cfg
            self.rotary_emb = LlamaRotaryEmbedding(config=cfg)

    model = _Model().to(torch.bfloat16)  # what create_hf_model does
    assert model.rotary_emb.inv_freq.dtype == torch.bfloat16, "precondition: the buffer must be bf16"

    # Far enough out that a ~4.4e-4 relative frequency error is visible. The error is proportional to
    # position, so a short prompt would hide it -- which is exactly how this survived.
    pos = torch.tensor([[40000]], dtype=torch.long)
    hidden = torch.zeros(1, 1, cfg.hidden_size, dtype=torch.bfloat16)

    cos_fix, sin_fix = reference_rope(model, hidden, pos)

    # Ground truth: an independently constructed fp32 table.
    gt = LlamaRotaryEmbedding(config=cfg).float()
    with torch.no_grad():
        cos_gt, sin_gt = gt(torch.zeros(1, 1, cfg.hidden_size), pos)

    err_fix = max((cos_fix - cos_gt).abs().max().item(), (sin_fix - sin_gt).abs().max().item())
    assert err_fix < 1e-5, f"reference_rope should match an fp32 table; max err {err_fix:.2e}"

    # The pre-fix path, asserted to be detectably wrong so this test cannot pass vacuously.
    old = deepcopy(model.rotary_emb).float()
    with torch.no_grad():
        cos_old, sin_old = old(torch.zeros(1, 1, cfg.hidden_size), pos)
    err_old = max((cos_old - cos_gt).abs().max().item(), (sin_old - sin_gt).abs().max().item())
    assert err_old > 1e-3, (
        f"the deepcopy(bf16).float() path should be visibly wrong at position {pos.item()} "
        f"(got {err_old:.2e}); if this fires, the test has stopped discriminating"
    )


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


def test_a_layer_that_computes_rope_internally_needs_no_rotary_emb(expect_error):
    """Guards the DeepSeek regression: an unconditional rope build breaks references that never
    wanted one.

    A transformers-5.x layer requires the caller to pass ``(cos, sin)``; a vendored layer such as
    DeepSeekV3's computes rope from ``position_ids`` itself and its model exposes no top-level
    ``rotary_emb``. Building the table for everything asserted with
    "DeepseekV3Model exposes no rotary_emb to build rope from" on precisely the models that did not
    need it, so the predicate must come from the layer's SIGNATURE, not from the model's attributes.
    """

    old_layer, new_layer = _OldStyleLayer(), _NewStyleLayer()
    assert layer_wants_position_embeddings(old_layer) is False
    assert layer_wants_position_embeddings(new_layer) is True

    # A model with NO rotary_emb, as DeepSeekV3Model has none. Binding kwargs for the old-style
    # layer must not reach reference_rope at all -- if it does, the assert there fires.
    class _ModelWithoutRotary(torch.nn.Module):
        pass

    # The regression itself: the run-level hoist must return None here, not raise. Before the fix it
    # called reference_rope unconditionally and died on the missing rotary_emb.
    model = _ModelWithoutRotary()
    model.layers = [old_layer]
    assert reference_position_embeddings(model, torch.zeros(1, 4, 8), torch.arange(4).unsqueeze(0), 1) is None

    # ...and a layer that DOES want them still gets a table built (here: absent rotary -> loud).
    model.layers = [new_layer]
    with expect_error(AssertionError, "exposes no rotary_emb"):
        reference_position_embeddings(model, torch.zeros(1, 4, 8), torch.arange(4).unsqueeze(0), 1)


def test_mscale_toggle_invalidates_the_chunked_reference_cache():
    """Guards the mscale half of `fix(mla): build the reference rope from the config...`.

    `mla_disable_yarn_mscale` scales softmax by ``mscale**2``, so a cache key that ignored it served
    a stale reference for the opposite setting -- valid-looking, silently wrong. Holding weights and
    hidden state constant, flipping only that flag must change the key.
    """
    from types import SimpleNamespace

    from models.demos.deepseek_v3_d_p.utils.chunked_prefill_utils import _ref_cache_key

    weights = {"w": torch.zeros(4, 4)}
    hidden = torch.zeros(4, 4)

    def cfg(disable_mscale):
        return SimpleNamespace(
            num_attention_heads=8,
            rms_norm_eps=1e-6,
            mla_use_nope=False,
            mla_use_output_gate=False,
            mla_disable_yarn_mscale=disable_mscale,
            rope_scaling=None,
        )

    off = _ref_cache_key(cfg(False), weights, hidden)
    on = _ref_cache_key(cfg(True), weights, hidden)
    assert off != on, f"mscale toggle must change the reference cache key, got {off} for both"
    # and the key must be stable for an unchanged config, or every run misses
    assert off == _ref_cache_key(cfg(False), weights, hidden)
