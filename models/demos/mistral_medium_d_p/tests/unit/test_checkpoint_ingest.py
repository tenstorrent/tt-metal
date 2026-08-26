# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""HOST-ONLY: checkpoint ingest (prefix strip, per-tensor fp8 dequant, Meta RoPE swizzle, config
guards). **No TT hardware needed.**

The key names, dtypes and scale shapes below are the REAL ones, read from
``mistralai/Mistral-Medium-3.5-128B``'s ``model.safetensors.index.json`` and shard headers:
layer tensors are ``F8_E4M3`` with a **scalar** (shape ``[]``) BF16 ``weight_scale_inv`` and a scalar
``activation_scale``; ``lm_head`` / ``embed_tokens`` / norms are BF16; everything text-side is
prefixed ``model.language_model.``.

Run:  pytest models/demos/mistral_medium_d_p/tests/unit/test_checkpoint_ingest.py
"""

import json
import os

import pytest
import torch

from models.demos.mistral_medium_d_p.tt.checkpoint import (
    assert_supported,
    dequantize_fp8,
    load_hf_config_dict,
    strip_multimodal_wrapper,
)

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "Mistral-Medium-3.5-128B")
HIDDEN, N_Q, N_KV, HEAD_DIM, FFN = 12288, 96, 8, 128, 28672


def _fp8_pair(shape, scale_value):
    """A realistic (fp8 weight, scalar bf16 weight_scale_inv) pair plus the value it decodes to."""
    ref = torch.randn(*shape) * 0.02
    q = (ref / scale_value).to(torch.float8_e4m3fn)
    return q, torch.tensor(scale_value, dtype=torch.bfloat16), (q.to(torch.float32) * scale_value).to(torch.bfloat16)


def test_strip_multimodal_wrapper_maps_real_keys():
    sd = {
        "model.language_model.embed_tokens.weight": torch.zeros(1),
        "model.language_model.norm.weight": torch.zeros(1),
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(1),
        "model.language_model.layers.87.mlp.down_proj.weight": torch.zeros(1),
        "lm_head.weight": torch.zeros(1),
        "model.vision_tower.transformer.layers.0.attention.q_proj.weight": torch.zeros(1),
        "model.multi_modal_projector.linear_1.weight": torch.zeros(1),
    }
    out = strip_multimodal_wrapper(sd)
    assert set(out) == {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.87.mlp.down_proj.weight",
        "lm_head.weight",
    }, "vision tower / projector must be dropped and the language_model wrapper collapsed to model."


def test_dequantize_fp8_per_tensor_roundtrip_and_sidecar_removal():
    q, scale, expected = _fp8_pair((N_KV * HEAD_DIM, HIDDEN), 0.031)
    sd = {
        "model.layers.0.self_attn.k_proj.weight": q,
        "model.layers.0.self_attn.k_proj.weight_scale_inv": scale,
        "model.layers.0.self_attn.k_proj.activation_scale": torch.tensor(0.5, dtype=torch.bfloat16),
        "model.layers.0.input_layernorm.weight": torch.ones(HIDDEN, dtype=torch.bfloat16),
        "lm_head.weight": torch.zeros(4, HIDDEN, dtype=torch.bfloat16),
    }
    out = dequantize_fp8(sd)

    assert set(out) == {
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "lm_head.weight",
    }, "weight_scale_inv / activation_scale must not reach the TT stack"
    w = out["model.layers.0.self_attn.k_proj.weight"]
    assert w.dtype == torch.bfloat16 and w.shape == (N_KV * HEAD_DIM, HIDDEN)
    torch.testing.assert_close(w, expected, rtol=0, atol=0)
    # Untouched bf16 tensors must be the same objects/values, not re-cast.
    assert out["lm_head.weight"].dtype == torch.bfloat16


def test_dequantized_weight_tracks_the_original_closely():
    """fp8 e4m3 has ~2 decimal digits; a correct per-tensor dequant should still be >0.99 correlated."""
    ref = torch.randn(512, 512) * 0.02
    scale = (ref.abs().max() / 448.0).item()  # e4m3 max magnitude
    q = (ref / scale).to(torch.float8_e4m3fn)
    out = dequantize_fp8({"w.weight": q, "w.weight_scale_inv": torch.tensor(scale, dtype=torch.bfloat16)})
    got = out["w.weight"].float().flatten()
    pcc = torch.corrcoef(torch.stack([got, ref.flatten()]))[0, 1].item()
    assert pcc > 0.999, f"per-tensor fp8 dequant correlation {pcc} — scale convention is probably inverted"


def test_blockwise_fp8_is_rejected_not_broadcast():
    """DeepSeek-style [N/128, K/128] scales must fail loud, not silently broadcast."""
    q = torch.zeros(256, 256).to(torch.float8_e4m3fn)
    with pytest.raises(NotImplementedError, match="per-tensor"):  # allow-pytest.raises: host-only, no root conftest
        dequantize_fp8({"w.weight": q, "w.weight_scale_inv": torch.ones(2, 2)})


def test_meta_qkv_swizzle_is_the_inverse_of_hf_rope():
    """The Meta swizzle + interleaved cos/sin must reproduce HF rotate_half + concat cos/sin."""
    from models.demos.mistral_medium_d_p.tt.rope_tables import build_hf_cos_sin, build_yarn_cos_sin
    from models.tt_transformers.tt.load_checkpoints import convert_hf_qkv_to_meta_format

    seq = 64
    q_hf = torch.randn(N_Q * HEAD_DIM, HIDDEN) * 0.02
    x = torch.randn(1, seq, HIDDEN) * 0.1
    cfg = load_hf_config_dict(CONFIG_DIR)
    kw = dict(
        rope_theta=cfg["rope_parameters"]["rope_theta"],
        yarn_factor=cfg["rope_parameters"]["factor"],
        yarn_orig_max_pos=cfg["rope_parameters"]["original_max_position_embeddings"],
        yarn_beta_fast=cfg["rope_parameters"]["beta_fast"],
        yarn_beta_slow=cfg["rope_parameters"]["beta_slow"],
    )
    cos_hf, sin_hf = build_hf_cos_sin(seq, HEAD_DIM, **kw)
    cos_meta, sin_meta = build_yarn_cos_sin(seq, HEAD_DIM, **kw)

    def rot_half(t):
        a, b = t[..., : t.shape[-1] // 2], t[..., t.shape[-1] // 2 :]
        return torch.cat([-b, a], dim=-1)

    def rot_interleaved(t):
        a, b = t[..., 0::2], t[..., 1::2]
        return torch.stack([-b, a], dim=-1).flatten(-2)

    q_ref = (x @ q_hf.t()).view(1, seq, N_Q, HEAD_DIM).transpose(1, 2)
    ref = q_ref * cos_hf + rot_half(q_ref) * sin_hf

    q_meta = convert_hf_qkv_to_meta_format({"q_proj.weight": q_hf}, HEAD_DIM)["q_proj.weight"]
    q_tt = (x @ q_meta.t()).view(1, seq, N_Q, HEAD_DIM).transpose(1, 2)
    got = q_tt * cos_meta + rot_interleaved(q_tt) * sin_meta

    # Meta interleaves the frequency pairs, so column 2i/2i+1 of `got` is column i / i+half of `ref`.
    ref_interleaved = torch.stack([ref[..., : HEAD_DIM // 2], ref[..., HEAD_DIM // 2 :]], dim=-1).flatten(-2)
    torch.testing.assert_close(got, ref_interleaved, rtol=2e-4, atol=2e-4)


def test_assert_supported_accepts_the_shipped_config():
    assert_supported(load_hf_config_dict(CONFIG_DIR))


@pytest.mark.parametrize(
    "override, match",
    [
        ({"rope_parameters": {"rope_type": "llama3", "factor": 8}}, "rope_type"),
        ({"rope_parameters": {"rope_type": "yarn", "llama_4_scaling_beta": 0.1}}, "llama_4_scaling_beta"),
        ({"sliding_window": 4096}, "sliding_window"),
        ({"hidden_act": "swigluoai"}, "hidden_act"),
        ({"attention_bias": True}, "attention_bias"),
        ({"tie_word_embeddings": True}, "tie_word_embeddings"),
        ({"num_attention_heads": 64}, "head_dim"),
    ],
)
def test_assert_supported_rejects_unimplemented_mechanisms(override, match):
    cfg = load_hf_config_dict(CONFIG_DIR)
    cfg.update(override)
    with pytest.raises(NotImplementedError, match=match):  # allow-pytest.raises: host-only, no root conftest
        assert_supported(cfg)


def test_bundled_config_matches_the_published_one():
    """Guard the flattened config against drift from the numbers verified upstream."""
    cfg = load_hf_config_dict(CONFIG_DIR)
    assert (cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"]) == (HIDDEN, FFN, 88)
    assert (cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]) == (N_Q, N_KV, HEAD_DIM)
    assert cfg["vocab_size"] == 131072 and cfg["max_position_embeddings"] == 262144
    assert cfg["rms_norm_eps"] == 1e-5 and cfg["sliding_window"] is None
    rp = cfg["rope_parameters"]
    assert (rp["rope_type"], rp["rope_theta"], rp["factor"]) == ("yarn", 1000000.0, 64.0)
    assert (rp["original_max_position_embeddings"], rp["beta_fast"], rp["beta_slow"]) == (4096, 4.0, 1.0)
    assert cfg["quantization_config"]["weight_block_size"] is None, "per-tensor fp8, not blockwise"
    with open(os.path.join(CONFIG_DIR, "config.json")) as f:
        assert "vision_config" not in json.load(f), "text backbone only"
