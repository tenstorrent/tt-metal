# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks for the MLA (DFlash2) drafter's prefill path: no device needed.

    pytest models/demos/deepseek_v3_d_p/tests/dflash_prefill/test_dflash_mla_host.py -svv

* config parsing: MLA detection and fail-closed ``num_target_layers``;
* the device rope table (``mla/rope.get_cos_sin_matrix`` on the drafter's hf config) equals TorchSpec's YaRN
  as blaze's golden computes it -- the device and decode must rotate k_pe identically;
* checkpoint loading keeps only the MLA prefill subset and normalizes ``context_norm`` to ``hidden_norm``.
"""

from types import SimpleNamespace

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.dflash_prefill.mla_latent import (
    context_latents,
    rope_interleaved,
    rope_tables,
    synthetic_prefill_state_dict,
)
from models.demos.deepseek_v3_d_p.tt.dflash_prefill.dflash_drafter_config import (
    DFlashDrafterConfig,
    kimi_k3_dflash2_mla_config,
)
from models.demos.deepseek_v3_d_p.tt.dflash_prefill.utils import build_drafter_rope_hf_config, load_drafter_state_dict


def _hf_dflash2(**overrides):
    """A config.json shaped as TorchSpec writes DFlash2 (MLA). Unverified against the real checkpoint."""
    c = dict(
        architectures=["DFlash2DraftModel"],
        attention_mode="mla",
        hidden_size=7168,
        num_attention_heads=64,
        num_key_value_heads=64,
        num_hidden_layers=5,
        num_target_layers=93,
        rms_norm_eps=1e-5,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rope_scaling={
            "rope_type": "yarn",
            "factor": 32.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "original_max_position_embeddings": 32768,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
        rope_theta=50000.0,
        dflash_config={"target_layer_ids": [19, 37, 66, 78, 90], "block_size": 8, "mask_token_id": 163592},
    )
    c.update(overrides)
    return SimpleNamespace(**c)


def test_mla_config_parses_dflash2():
    cfg = DFlashDrafterConfig.from_hf_config(_hf_dflash2())
    assert cfg.is_mla and cfg.latent_dim == 576 and cfg.rope_dim == 64
    assert cfg.target_layer_ids == (19, 37, 66, 78, 90)
    assert cfg.num_target_layers == 93 and cfg.num_hidden_layers == 5
    assert (cfg.rope_theta, cfg.rope_factor, cfg.rope_orig_max_pos) == (50000.0, 32.0, 32768)
    assert cfg.rope_yarn_amplitude == 1.0
    assert cfg == kimi_k3_dflash2_mla_config(), "the preset and a TorchSpec-shaped config.json must agree"


@pytest.mark.parametrize(
    "drop, arch",
    [("attention_mode", ["DFlash2DraftModel"]), ("attention_mode", ["SomethingElse"])],
    ids=["by-architecture", "by-kv_lora_rank"],
)
def test_mla_mode_detection_fallbacks(drop, arch):
    c = _hf_dflash2(architectures=arch)
    delattr(c, drop)
    assert DFlashDrafterConfig.hf_attention_mode(c) == "mla"


def test_mla_num_target_layers_is_fail_closed():
    c = _hf_dflash2()
    delattr(c, "num_target_layers")
    assert DFlashDrafterConfig.from_hf_config(c).num_target_layers is None


def test_gqa_config_unchanged():
    c = SimpleNamespace(
        architectures=["DFlashDraftModel"],
        hidden_size=7168,
        head_dim=128,
        num_attention_heads=64,
        num_key_value_heads=8,
        num_hidden_layers=6,
        num_target_layers=61,
        rms_norm_eps=1e-5,
        rope_scaling={"factor": 64.0, "original_max_position_embeddings": 4096},
        rope_theta=50000.0,
        dflash_config={"target_layer_ids": [1, 12, 24, 35, 47, 58]},
    )
    cfg = DFlashDrafterConfig.from_hf_config(c)
    assert not cfg.is_mla and cfg.rope_dim == 128 and cfg.num_key_value_heads == 8


def test_device_rope_table_matches_torchspec_yarn():
    """``rotary_embedding_indexed`` consumes ``get_cos_sin_matrix(interleave=True)``; it must equal the decode
    golden's interleaved-pair YaRN table (same frequencies, unit amplitude) at every position."""
    from models.demos.deepseek_v3_d_p.tt.mla.rope import get_cos_sin_matrix

    cfg = kimi_k3_dflash2_mla_config()
    n = 70_000  # past original_max_position_embeddings, where YaRN's ramp matters
    hf = build_drafter_rope_hf_config(cfg, max_seq_len=n)
    cos_dev, sin_dev = get_cos_sin_matrix(hf, interleave=True, num_positions=n)
    cos_ref, sin_ref = rope_tables(torch.arange(n), cfg)
    torch.testing.assert_close(cos_dev[0, 0, :n].float(), cos_ref, atol=2e-5, rtol=0)
    torch.testing.assert_close(sin_dev[0, 0, :n].float(), sin_ref, atol=2e-5, rtol=0)


def test_rope_is_linear_so_amplitude_folds_into_kv_a():
    """``TtDFlashMlaDrafter`` folds a non-unit YaRN amplitude into kv_a's k_pe rows; that is exact only because
    rope(s * x) == s * rope(x). Pin it, since a future rope with a bias/clamp would silently break the fold."""
    cfg = kimi_k3_dflash2_mla_config(rope_mscale=0.707)
    cos, sin = rope_tables(torch.arange(64), cfg)
    x = torch.randn(64, 64)
    unit = kimi_k3_dflash2_mla_config()
    cos1, sin1 = rope_tables(torch.arange(64), unit)
    torch.testing.assert_close(
        rope_interleaved(x, cos, sin), rope_interleaved(x * cfg.rope_yarn_amplitude, cos1, sin1), atol=1e-5, rtol=0
    )


def test_reference_latent_shapes_and_position_offset():
    """A chunk at offset ``start`` matches the same rows of a from-zero build (latent is token-parallel)."""
    cfg = kimi_k3_dflash2_mla_config(hidden_size=256, num_hidden_layers=2, target_layer_ids=(19, 37))
    sd = synthetic_prefill_state_dict(cfg)
    taps = [torch.randn(96, cfg.hidden_size) for _ in cfg.target_layer_ids]
    full = context_latents(sd, taps, cfg)
    assert full.shape == (2, 96, 576)
    tail = context_latents(sd, [t[64:] for t in taps], cfg, start=64)
    torch.testing.assert_close(tail, full[:, 64:], atol=1e-4, rtol=0)


def test_load_state_dict_keeps_mla_subset_and_renames_context_norm(tmp_path):
    from safetensors.torch import save_file

    cfg = kimi_k3_dflash2_mla_config(hidden_size=64, num_hidden_layers=2, target_layer_ids=(19, 37))
    sd = synthetic_prefill_state_dict(cfg)
    sd["context_norm.weight"] = sd.pop("hidden_norm.weight")
    sd["layers.0.self_attn.q_a_proj.weight"] = torch.zeros(8, 64)  # decode-only: must not be loaded
    sd["layers.0.attention_conv.base_kernel"] = torch.zeros(2, 2, 64)
    save_file({k: v.contiguous() for k, v in sd.items()}, str(tmp_path / "model.safetensors"))

    tail = load_drafter_state_dict(str(tmp_path), build_kv_tail=True, attention_mode="mla")
    assert set(tail) == {
        "fc.weight",
        "hidden_norm.weight",
        *(f"layers.{i}.self_attn.{n}.weight" for i in range(2) for n in ("kv_a_proj_with_mqa", "kv_a_layernorm")),
    }
    assert set(load_drafter_state_dict(str(tmp_path), build_kv_tail=False, attention_mode="mla")) == {"fc.weight"}
