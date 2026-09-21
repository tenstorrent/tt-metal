# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""P1: the checkpoint loader — resolution, key mapping, and the no-dequantizer guard. Host only.

There is no dequantization test here because there is nothing to dequantize: Qwen3.8-27B ships
unquantized bf16. The corresponding risk is the opposite one — a *future* quantized checkpoint
being read as if its packed blocks were values — so ``assert_unquantized`` is what gets tested.

The rest is key mapping. The published checkpoint is a VL package: the text tower is under
``model.language_model.``, the vision tower under ``model.visual.``, the MTP head under ``mtp.``,
and ``lm_head.weight`` sits at the top level. Loading a vision or MTP tensor into a text-tower
slot would not raise anywhere — the shapes are plausible — so the filter is pinned by name.
"""

from __future__ import annotations

import json

import pytest
import torch

from models.demos.qwen_3_8_27b_d_p.reference.config import Qwen35TextConfig
from models.demos.qwen_3_8_27b_d_p.reference.modeling import Qwen35TextModel
from models.demos.qwen_3_8_27b_d_p.tt.weights import (
    DROPPED_PREFIXES,
    TEXT_PREFIX,
    assert_unquantized,
    load_text_backbone_state_dict,
    resolve_checkpoint_path,
)

CHECKPOINT = resolve_checkpoint_path(required=False)
requires_checkpoint = pytest.mark.skipif(CHECKPOINT is None, reason="no Qwen3.8-27B checkpoint on this host")


@requires_checkpoint
def test_checkpoint_is_unquantized():
    """The loader has no dequantizer, so a quantized checkpoint must be a hard error, not a
    silently wrong read."""
    assert_unquantized(CHECKPOINT)


@requires_checkpoint
def test_index_carries_the_expected_key_families():
    with open(CHECKPOINT / "model.safetensors.index.json") as f:
        keys = list(json.load(f)["weight_map"])
    assert any(k.startswith(TEXT_PREFIX) for k in keys), "no text-tower keys in the index"
    assert any(k.startswith("model.visual.") for k in keys), "expected a vision tower to drop"
    assert any(k.startswith("mtp.") for k in keys), "expected an MTP head to drop"
    assert "lm_head.weight" in keys


@requires_checkpoint
def test_loader_keys_match_the_reference(monkeypatch):
    """A 4-layer slice, loaded and fed to ``load_state_dict`` — the names have to line up exactly.

    ``strict``-style checking here is the point: the reference and the TT modules read the same
    key names, so this one assertion covers both.
    """
    cfg = Qwen35TextConfig.from_json().reduced(4)
    state_dict = load_text_backbone_state_dict(CHECKPOINT, layers=range(4), dtype=torch.float16)

    assert not any(k.startswith(DROPPED_PREFIXES) for k in state_dict), "vision or MTP tensors leaked through"
    assert "embed_tokens.weight" in state_dict
    assert "norm.weight" in state_dict
    assert "lm_head.weight" in state_dict
    assert not any(k.startswith("layers.4.") for k in state_dict), "loaded layers outside the requested slice"

    with torch.device("meta"):
        model = Qwen35TextModel(cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    assert not missing, f"the checkpoint is missing reference parameters: {missing[:5]}"
    assert not unexpected, f"the checkpoint has parameters the reference does not: {unexpected[:5]}"


@requires_checkpoint
def test_loaded_shapes_match_the_config():
    """Dimension sanity against the vendored config — catches a checkpoint/config mismatch before
    it becomes an unexplained PCC number."""
    cfg = Qwen35TextConfig.from_json()
    sd = load_text_backbone_state_dict(CHECKPOINT, layers=range(4), dtype=torch.bfloat16)
    assert sd["embed_tokens.weight"].shape == (cfg.vocab_size, cfg.hidden_size)
    assert sd["lm_head.weight"].shape == (cfg.vocab_size, cfg.hidden_size)
    # layer 0 is Gated DeltaNet, layer 3 is full attention.
    assert sd["layers.0.linear_attn.in_proj_qkv.weight"].shape == (cfg.gdn_conv_dim, cfg.hidden_size)
    assert sd["layers.0.linear_attn.conv1d.weight"].shape == (cfg.gdn_conv_dim, 1, cfg.linear_conv_kernel_dim)
    assert sd["layers.0.linear_attn.A_log"].shape == (cfg.linear_num_value_heads,)
    # q_proj is TWICE the usual width — the output gate lives in the second half of each head.
    assert sd["layers.3.self_attn.q_proj.weight"].shape == (
        cfg.num_attention_heads * cfg.head_dim * 2,
        cfg.hidden_size,
    )
    assert sd["layers.3.self_attn.k_proj.weight"].shape == (
        cfg.num_key_value_heads * cfg.head_dim,
        cfg.hidden_size,
    )
    assert sd["layers.3.self_attn.q_norm.weight"].shape == (cfg.head_dim,)
    assert sd["layers.0.mlp.gate_proj.weight"].shape == (cfg.intermediate_size, cfg.hidden_size)
    assert all(t.dtype == torch.bfloat16 for t in sd.values()), "the single dtype exit did not fire"
