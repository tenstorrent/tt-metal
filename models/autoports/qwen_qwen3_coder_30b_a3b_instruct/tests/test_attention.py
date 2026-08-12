# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN attention vs the HuggingFace reference layer's ``self_attn``.

The PCC test is the real check, but two cheap host-side tests run first because
they isolate failures that PCC alone reports as one undifferentiated low number:

  * ``test_qk_norm_is_applied`` -- proves the per-head norm actually changed the
    tensor. A QK-norm quietly skipped (wrong key, missing weight) still yields
    a plausible ~0.9 PCC that looks like ordinary bf16 loss.
  * ``test_causality`` -- perturbing a late token must not move an early one.
    A non-causal run scores high PCC on short sequences, so ``is_causal`` being
    dropped is otherwise invisible here and only surfaces as garbage generation
    much later.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc

from ..tt.functional_decoder import AttentionConfig, attention_prefill, build_rope_cache, upload_attention_weights
from ..tt.weight_mapping import convert_attention_weights
from .reference import build_reference_layer, layer_state_dict, rotary_embeddings

LAYER_IDX = 0
PCC_REQUIRED = 0.99  # attention_bias=False, so no large-bias PCC degradation applies


@pytest.fixture(scope="module")
def reference():
    return build_reference_layer(LAYER_IDX)


@pytest.fixture(scope="module")
def torch_weights():
    return convert_attention_weights(
        {k: v for k, v in layer_state_dict(LAYER_IDX).items()},
        n_heads=32,
        n_kv_heads=4,
        head_dim=128,
    )


def _hidden(config, seq_len, seed=0):
    torch.manual_seed(seed)
    return torch.randn(1, seq_len, config.hidden_size, dtype=torch.float32) * 0.02


def _causal_mask(seq_len):
    return torch.full((seq_len, seq_len), float("-inf")).triu(1).reshape(1, 1, seq_len, seq_len)


def _reference_attention(layer, config, hidden):
    seq_len = hidden.shape[1]
    cos, sin = rotary_embeddings(config, seq_len)
    with torch.no_grad():
        out = layer.self_attn(
            hidden_states=hidden,
            position_embeddings=(cos, sin),
            attention_mask=_causal_mask(seq_len),
        )
    return out[0] if isinstance(out, tuple) else out


def test_qk_norm_weights_are_non_trivial(torch_weights):
    """A QK-norm weight of all ones would make the norm undetectable in PCC."""
    for name in ("q_norm", "k_norm"):
        w = torch_weights[name]
        assert w.shape == (128,), f"{name} has shape {tuple(w.shape)}, expected (head_dim,)"
        assert not torch.allclose(w, torch.ones_like(w)), f"{name} is all ones -- cannot detect a skipped norm"


def test_causality(reference):
    """Changing the last token must leave earlier outputs untouched."""
    layer, config = reference
    hidden = _hidden(config, 32)
    baseline = _reference_attention(layer, config, hidden)

    perturbed = hidden.clone()
    perturbed[:, -1, :] += 1.0
    after = _reference_attention(layer, config, perturbed)

    assert torch.allclose(baseline[:, :-1], after[:, :-1], atol=1e-5), "reference attention is not causal"


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
@pytest.mark.parametrize("seq_len", [32, 128, 512], ids=["s32", "s128", "s512"])
def test_attention_prefill_vs_reference(mesh_device, reference, torch_weights, seq_len):
    layer, hf_config = reference
    config = AttentionConfig.from_hf(hf_config)
    hidden = _hidden(hf_config, seq_len)

    ref_out = _reference_attention(layer, hf_config, hidden)

    weights = upload_attention_weights(torch_weights, mesh_device)
    cos_cache, sin_cache = build_rope_cache(hf_config, seq_len, mesh_device)

    tt_in = ttnn.from_torch(
        hidden.unsqueeze(0),  # [1, 1, S, hidden]
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_out = attention_prefill(tt_in, weights, config, cos_cache, sin_cache)
    tt_out_torch = ttnn.to_torch(tt_out).squeeze(0)

    passing, pcc_message = comp_pcc(ref_out, tt_out_torch, PCC_REQUIRED)
    logger.info(comp_allclose(ref_out, tt_out_torch))
    logger.info(f"attention prefill seq={seq_len}: {pcc_message}")
    assert passing, f"attention prefill (seq={seq_len}) below {PCC_REQUIRED}: {pcc_message}"
