# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""CPU-only smoke tests for the V2 ``load_checkpoints.py`` Qwen3.6 branch.

These tests verify key-name plumbing only — no real weight tensors are loaded.
Each test builds a synthetic state-dict whose values are zero-cost ``torch.empty``
tensors with the canonical shapes from the HF safetensors index, then asserts
the post-conversion key set matches what V2-4 (``llama_attention.py``) and V2-5
(``qwen36_delta_attention.py``) expect to consume.

Run:

    pytest models/demos/qwen3_6_galaxy_v2/tests/test_load_checkpoints.py -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from models.demos.qwen3_6_galaxy_v2.tt.load_checkpoints import (
    _is_qwen36_state_dict,
    convert_hf_to_meta,
    get_qwen36_linear_attention_pattern,
    standardize_hf_keys,
    standardize_hf_keys_qwen36,
)

_HF_SNAPSHOT = Path(
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/" "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
    )
)
_INDEX_PATH = _HF_SNAPSHOT / "model.safetensors.index.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _load_qwen36_index_keys():
    """Read just the key set from the HF safetensors index — no tensor data."""
    if not _INDEX_PATH.exists():
        pytest.skip(f"Qwen3.6 HF snapshot index not found at {_INDEX_PATH}")
    with open(_INDEX_PATH) as f:
        idx = json.load(f)
    return list(idx["weight_map"].keys())


def _synthesize_state_dict(keys):
    """Build a key-only state-dict with tiny placeholder tensors.

    We don't need real shapes for the key-plumbing tests because none of
    the load_checkpoints helpers under test inspect tensor shapes (except
    ``convert_hf_qkv_to_meta_format``, which we deliberately skip for qwen3.6
    via the ``is_qwen36`` branch).
    """
    return {k: torch.empty(1) for k in keys}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_detect_qwen36_state_dict():
    """``_is_qwen36_state_dict`` should pick up the ``model.language_model.`` prefix."""
    qwen36_sd = {"model.language_model.embed_tokens.weight": torch.empty(1)}
    llama_sd = {"model.embed_tokens.weight": torch.empty(1)}
    assert _is_qwen36_state_dict(qwen36_sd)
    assert not _is_qwen36_state_dict(llama_sd)


@pytest.mark.cpu_only
def test_standardize_drops_visual_and_mtp():
    """VLM (vision tower) and MTP keys must be stripped before key-rename."""
    raw = {
        "model.language_model.embed_tokens.weight": torch.empty(1),
        "model.language_model.norm.weight": torch.empty(1),
        "model.language_model.layers.3.self_attn.q_proj.weight": torch.empty(1),
        "model.visual.blocks.0.attn.qkv.weight": torch.empty(1),
        "model.visual.patch_embed.proj.weight": torch.empty(1),
        "mtp.fc.weight": torch.empty(1),
        "lm_head.weight": torch.empty(1),
    }
    out = standardize_hf_keys(raw)
    assert not any("visual" in k for k in out)
    assert not any(k.startswith("mtp.") for k in out)
    # Prefix-rewrite happened.
    assert "model.embed_tokens.weight" in out
    assert "model.norm.weight" in out
    assert "model.layers.3.self_attn.q_proj.weight" in out
    # lm_head.weight is preserved verbatim (no ``model.`` prefix).
    assert "lm_head.weight" in out


@pytest.mark.cpu_only
def test_standardize_qwen36_does_not_alias_lm_head_when_present():
    """Qwen3.6 has tie_word_embeddings=False — keep checkpoint's lm_head."""
    raw = {
        "model.language_model.embed_tokens.weight": torch.tensor([1.0]),
        "lm_head.weight": torch.tensor([2.0]),
    }
    out = standardize_hf_keys_qwen36(raw)
    assert torch.equal(out["lm_head.weight"], torch.tensor([2.0]))
    assert torch.equal(out["model.embed_tokens.weight"], torch.tensor([1.0]))


@pytest.mark.cpu_only
def test_layer3_full_attention_keys_round_trip():
    """For a ``full_attention`` layer, the meta key set must include wq/wk/wv/wo
    and q_norm/k_norm under the ``layers.3.attention.*`` namespace."""
    raw_keys = [
        "model.language_model.layers.3.input_layernorm.weight",
        "model.language_model.layers.3.post_attention_layernorm.weight",
        "model.language_model.layers.3.self_attn.q_proj.weight",
        "model.language_model.layers.3.self_attn.k_proj.weight",
        "model.language_model.layers.3.self_attn.v_proj.weight",
        "model.language_model.layers.3.self_attn.o_proj.weight",
        "model.language_model.layers.3.self_attn.q_norm.weight",
        "model.language_model.layers.3.self_attn.k_norm.weight",
        "model.language_model.layers.3.mlp.gate_proj.weight",
        "model.language_model.layers.3.mlp.up_proj.weight",
        "model.language_model.layers.3.mlp.down_proj.weight",
    ]
    sd = _synthesize_state_dict(raw_keys)
    sd = standardize_hf_keys(sd)
    # Pass ``is_qwen36=True`` explicitly: in this isolated full_attention-only
    # scenario there are no ``.linear_attn.*`` keys, so the auto-detect after
    # the standardize step would fall back to the llama permute path.
    sd = convert_hf_to_meta(sd, head_dim=256, is_qwen36=True)
    expected = {
        "layers.3.attention_norm.weight",
        "layers.3.ffn_norm.weight",
        "layers.3.attention.wq.weight",
        "layers.3.attention.wk.weight",
        "layers.3.attention.wv.weight",
        "layers.3.attention.wo.weight",
        "layers.3.attention.q_norm.weight",
        "layers.3.attention.k_norm.weight",
        "layers.3.feed_forward.w1.weight",
        "layers.3.feed_forward.w3.weight",
        "layers.3.feed_forward.w2.weight",
    }
    assert expected == set(sd.keys()), f"diff = {expected ^ set(sd.keys())}"


@pytest.mark.cpu_only
def test_layer0_linear_attention_keys_round_trip():
    """For a ``linear_attention`` layer, the meta key set must include all 9
    DeltaNet sub-keys under ``layers.0.linear_attn.*`` plus MLP and norms."""
    raw_keys = [
        "model.language_model.layers.0.input_layernorm.weight",
        "model.language_model.layers.0.post_attention_layernorm.weight",
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.0.linear_attn.in_proj_z.weight",
        "model.language_model.layers.0.linear_attn.in_proj_a.weight",
        "model.language_model.layers.0.linear_attn.in_proj_b.weight",
        "model.language_model.layers.0.linear_attn.conv1d.weight",
        "model.language_model.layers.0.linear_attn.A_log",
        "model.language_model.layers.0.linear_attn.dt_bias",
        "model.language_model.layers.0.linear_attn.norm.weight",
        "model.language_model.layers.0.linear_attn.out_proj.weight",
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.mlp.up_proj.weight",
        "model.language_model.layers.0.mlp.down_proj.weight",
    ]
    sd = _synthesize_state_dict(raw_keys)
    sd = standardize_hf_keys(sd)
    sd = convert_hf_to_meta(sd, head_dim=256)
    expected = {
        "layers.0.attention_norm.weight",
        "layers.0.ffn_norm.weight",
        "layers.0.linear_attn.in_proj_qkv.weight",
        "layers.0.linear_attn.in_proj_z.weight",
        "layers.0.linear_attn.in_proj_a.weight",
        "layers.0.linear_attn.in_proj_b.weight",
        "layers.0.linear_attn.conv1d.weight",
        "layers.0.linear_attn.A_log",
        "layers.0.linear_attn.dt_bias",
        "layers.0.linear_attn.norm.weight",
        "layers.0.linear_attn.out_proj.weight",
        "layers.0.feed_forward.w1.weight",
        "layers.0.feed_forward.w3.weight",
        "layers.0.feed_forward.w2.weight",
    }
    assert expected == set(sd.keys()), f"diff = {expected ^ set(sd.keys())}"


@pytest.mark.cpu_only
def test_top_level_keys_round_trip():
    """Embedding, norm, and lm_head must map onto the meta-style top-level keys."""
    raw_keys = [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    ]
    sd = _synthesize_state_dict(raw_keys)
    sd = standardize_hf_keys(sd)
    # Force qwen36 routing — top-level keys carry no namespace signal after the
    # ``model.language_model.`` prefix has been stripped by standardize.
    sd = convert_hf_to_meta(sd, head_dim=256, is_qwen36=True)
    assert set(sd.keys()) == {
        "tok_embeddings.weight",
        "norm.weight",
        "output.weight",
    }


@pytest.mark.cpu_only
def test_full_hf_index_produces_no_unknown_layer_keys():
    """End-to-end: feed every LM key from the real HF index through standardize +
    convert; every ``layers.{i}.*`` HF key must land in the meta state-dict.

    Also asserts the qwen3.6 detector handles the genuine layer-0 / layer-3
    structure: layer 0 (DeltaNet) yields a ``linear_attn`` namespace, layer 3
    (full_attention) yields an ``attention`` namespace.
    """
    keys = _load_qwen36_index_keys()
    # Restrict to the LM tower; the vision tower and MTP are dropped by
    # ``standardize_hf_keys`` and we don't want them counted.
    lm_keys = [k for k in keys if k.startswith("model.language_model.") or k == "lm_head.weight"]
    sd = _synthesize_state_dict(lm_keys)
    sd = standardize_hf_keys(sd)
    sd = convert_hf_to_meta(sd, head_dim=256)

    # Every output key should be in the canonical meta layout.
    for k in sd:
        assert k.startswith(("tok_embeddings", "norm", "output", "layers.")), k

    # Spot-check layer 0 (DeltaNet) and layer 3 (full_attention).
    layer0 = {k for k in sd if k.startswith("layers.0.")}
    layer3 = {k for k in sd if k.startswith("layers.3.")}
    assert "layers.0.linear_attn.in_proj_qkv.weight" in layer0
    assert "layers.0.linear_attn.A_log" in layer0
    assert "layers.0.linear_attn.dt_bias" in layer0
    assert not any("attention.w" in k for k in layer0), layer0
    assert "layers.3.attention.wq.weight" in layer3
    assert "layers.3.attention.q_norm.weight" in layer3
    assert not any("linear_attn" in k for k in layer3), layer3


@pytest.mark.cpu_only
def test_linear_attention_pattern_matches_config():
    """The pattern helper must return a length-64 list with 48 linear / 16 full."""
    if not _HF_SNAPSHOT.exists():
        pytest.skip(f"Qwen3.6 HF snapshot not found at {_HF_SNAPSHOT}")
    pattern = get_qwen36_linear_attention_pattern(str(_HF_SNAPSHOT))
    assert len(pattern) == 64
    assert pattern.count("linear_attention") == 48
    assert pattern.count("full_attention") == 16
    # Layer 0 is DeltaNet, layer 3 is full_attention (every 4th layer).
    assert pattern[0] == "linear_attention"
    assert pattern[3] == "full_attention"
    assert pattern[7] == "full_attention"


@pytest.mark.cpu_only
def test_llama_path_untouched():
    """A llama-style state-dict (no ``model.language_model.`` prefix) must take
    the legacy code path: tied lm_head when missing, standard meta key rename."""
    raw = {
        "model.embed_tokens.weight": torch.tensor([1.0]),
        "model.norm.weight": torch.empty(1),
        "model.layers.0.input_layernorm.weight": torch.empty(1),
        "model.layers.0.post_attention_layernorm.weight": torch.empty(1),
    }
    out = standardize_hf_keys(raw)
    # lm_head was tied to embed_tokens.
    assert "lm_head.weight" in out
    assert torch.equal(out["lm_head.weight"], torch.tensor([1.0]))
    # No qwen3.6-specific keys leaked into the llama path.
    assert not any("language_model" in k for k in out)
