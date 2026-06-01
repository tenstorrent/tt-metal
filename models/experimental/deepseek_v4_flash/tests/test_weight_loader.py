"""Unit tests for the DeepSeek-V4-Flash safetensors weight loader.

These tests exercise the loader against the real on-disk checkpoint at the
default HuggingFace cache path. They are skipped automatically when the
checkpoint is not present, so they're safe to run in environments where the
weights have not been downloaded.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models.experimental.deepseek_v4_flash.tt.weight_loader import (
    DeepseekV4WeightLoader,
    hf_to_checkpoint_name,
    resolve_snapshot_dir,
)


DEFAULT_MODEL_DIR = Path("/home/ttuser/models/hub/models--deepseek-ai--DeepSeek-V4-Flash")


# ---------------------------------------------------------------------- #
# Pure-python name mapping: no weights required
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "hf_name, expected",
    [
        ("model.embed_tokens.weight", "embed.weight"),
        ("embed_tokens.weight", "embed.weight"),
        ("lm_head.weight", "head.weight"),
        ("model.norm.weight", "norm.weight"),
        ("model.layers.0.input_layernorm.weight", "layers.0.attn_norm.weight"),
        ("model.layers.7.post_attention_layernorm.weight", "layers.7.ffn_norm.weight"),
        ("model.layers.3.self_attn.q_a_proj.weight", "layers.3.attn.wq_a.weight"),
        ("model.layers.3.self_attn.q_a_norm.weight", "layers.3.attn.q_norm.weight"),
        ("model.layers.3.self_attn.kv_proj.scale", "layers.3.attn.wkv.scale"),
        ("model.layers.3.self_attn.o_b_proj.weight", "layers.3.attn.wo_b.weight"),
        ("model.layers.3.self_attn.sinks", "layers.3.attn.attn_sink"),
        ("model.layers.5.attn_hc.fn", "layers.5.hc_attn_fn"),
        ("model.layers.5.ffn_hc.scale", "layers.5.hc_ffn_scale"),
        ("model.layers.2.mlp.gate.weight", "layers.2.ffn.gate.weight"),
        ("model.layers.2.mlp.gate.e_score_correction_bias", "layers.2.ffn.gate.bias"),
        ("model.layers.2.mlp.experts.17.gate_proj.weight", "layers.2.ffn.experts.17.w1.weight"),
        ("model.layers.2.mlp.experts.17.down_proj.scale", "layers.2.ffn.experts.17.w2.scale"),
        ("model.layers.2.mlp.experts.17.up_proj.weight", "layers.2.ffn.experts.17.w3.weight"),
        ("model.layers.2.mlp.shared_experts.gate_proj.weight", "layers.2.ffn.shared_experts.w1.weight"),
        ("model.layers.6.self_attn.compressor.kv_proj.weight", "layers.6.attn.compressor.wkv.weight"),
        ("model.layers.6.self_attn.compressor.position_bias", "layers.6.attn.compressor.ape"),
        ("model.layers.6.self_attn.compressor.indexer.q_b_proj.weight", "layers.6.attn.indexer.wq_b.weight"),
        ("hc_head.hc_fn", "hc_head_fn"),
        # Unknown names pass through untouched
        ("some.unmapped.parameter", "some.unmapped.parameter"),
    ],
)
def test_hf_to_checkpoint_name(hf_name: str, expected: str) -> None:
    assert hf_to_checkpoint_name(hf_name) == expected


# ---------------------------------------------------------------------- #
# Disk-backed tests: require the checkpoint to be on the box
# ---------------------------------------------------------------------- #
def _checkpoint_available() -> bool:
    try:
        resolve_snapshot_dir(DEFAULT_MODEL_DIR)
    except FileNotFoundError:
        return False
    return True


pytestmark_needs_ckpt = pytest.mark.skipif(
    not _checkpoint_available(),
    reason=f"V4-Flash checkpoint not found under {DEFAULT_MODEL_DIR}",
)


@pytest.fixture(scope="module")
def loader() -> DeepseekV4WeightLoader:
    return DeepseekV4WeightLoader(DEFAULT_MODEL_DIR)


@pytestmark_needs_ckpt
def test_loader_discovers_expected_tensors(loader: DeepseekV4WeightLoader) -> None:
    keys = set(loader.keys())
    # Sanity bounds: V4-Flash has 43 decoder layers + 1 MTP layer + embed/head/norm
    # plus 256 experts per MoE layer, so we expect well over a thousand tensors.
    assert len(keys) > 1000, f"Only {len(keys)} tensors discovered"
    # A few well-known names must be present.
    for required in (
        "embed.weight",
        "norm.weight",
        "head.weight",
        "layers.0.attn_norm.weight",
        "layers.0.attn.wq_a.weight",
    ):
        assert required in keys, f"Missing expected tensor: {required}"


@pytestmark_needs_ckpt
def test_get_meta_does_not_load_data(loader: DeepseekV4WeightLoader) -> None:
    dtype, shape = loader.get_meta("embed_tokens.weight")
    assert dtype == "BF16"
    vocab_size, hidden_size = shape
    assert vocab_size == 129280
    assert hidden_size == 4096


@pytestmark_needs_ckpt
def test_loads_embed_tokens_weight(loader: DeepseekV4WeightLoader) -> None:
    tensor = loader.get_tensor("embed_tokens.weight")
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (129280, 4096)
    assert tensor.dtype == torch.bfloat16
    # The embedding should contain real values, not be all zeros.
    assert tensor.abs().float().sum().item() > 0.0


@pytestmark_needs_ckpt
def test_hf_and_native_names_load_same_tensor(loader: DeepseekV4WeightLoader) -> None:
    via_hf = loader.get_tensor("model.embed_tokens.weight")
    via_native = loader.get_tensor("embed.weight", translate=False)
    assert torch.equal(via_hf, via_native)


@pytestmark_needs_ckpt
def test_get_scale_returns_none_for_unquantized(loader: DeepseekV4WeightLoader) -> None:
    # The embedding is stored in bf16, no companion .scale tensor exists.
    assert loader.get_scale("embed_tokens.weight") is None
    # Norm weights are unquantized too.
    assert loader.get_scale("model.norm.weight") is None


@pytestmark_needs_ckpt
def test_get_scale_returns_tensor_for_quantized_projection(
    loader: DeepseekV4WeightLoader,
) -> None:
    weight = loader.get_tensor("model.layers.0.self_attn.q_a_proj.weight")
    scale = loader.get_scale("model.layers.0.self_attn.q_a_proj.weight")
    assert scale is not None
    assert isinstance(scale, torch.Tensor)
    # The fp8 weight uses a 128x128 per-block ue8m0 scale, so the scale tensor
    # is roughly weight.shape // (128, 128) (rounded up). Just check the
    # leading dim is consistent and the scale is non-empty.
    assert scale.numel() > 0
    assert scale.shape[0] == (weight.shape[0] + 127) // 128


@pytestmark_needs_ckpt
def test_has_round_trips(loader: DeepseekV4WeightLoader) -> None:
    assert loader.has("model.embed_tokens.weight")
    assert loader.has("embed.weight", translate=False)
    assert not loader.has("model.embed_tokens.does_not_exist")


@pytestmark_needs_ckpt
def test_missing_tensor_raises(loader: DeepseekV4WeightLoader) -> None:
    with pytest.raises(KeyError):
        loader.get_tensor("model.this.is.not.a.real.tensor")
