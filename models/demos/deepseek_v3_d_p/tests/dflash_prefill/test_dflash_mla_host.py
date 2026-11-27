# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks for the MLA (DFlash2) drafter's prefill path: no device needed.

    pytest models/demos/deepseek_v3_d_p/tests/dflash_prefill/test_dflash_mla_host.py -svv

* config parsing: MLA detection and fail-closed ``num_target_layers``;
* the device rope table (``mla/rope.get_cos_sin_matrix`` on the drafter's hf config) equals TorchSpec's YaRN
  as blaze's golden computes it -- the device and decode must rotate k_pe identically;
* checkpoint loading keeps only the MLA prefill subset and normalizes ``context_norm`` to ``hidden_norm``;
* Kimi-K3.1's released drafter (``RadixArk/Kimi-K3.1-Dflash2-Preview``): its ``config.json`` as shipped, six taps
  feeding five layers, and -- when the checkpoint is on disk -- its real key names and shapes.
"""

import json
import os
import struct

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
    drafter_context_window,
    kimi_k3_dflash2_mla_config,
    kimi_k31_dflash2_mla_config,
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


@pytest.mark.parametrize("preset", [kimi_k3_dflash2_mla_config, kimi_k31_dflash2_mla_config], ids=["k3", "k3_1"])
def test_device_rope_table_matches_torchspec_yarn(preset):
    """``rotary_embedding_indexed`` consumes ``get_cos_sin_matrix(interleave=True)``; it must equal the decode
    golden's interleaved-pair YaRN table (same frequencies, unit amplitude) at every position."""
    from models.demos.deepseek_v3_d_p.tt.mla.rope import get_cos_sin_matrix

    cfg = preset()
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


# ---- Kimi-K3.1: RadixArk/Kimi-K3.1-Dflash2-Preview ------------------------------------------------------------
# The checkpoint's config.json, verbatim for every key the prefill path reads (and the ones that say what it must
# NOT assume: six taps for five layers, 96 heads, `head_dim` 64 beside `v_head_dim` 128, `rope_theta` at the top
# level but absent from `rope_scaling`).
K31_DFLASH2_CONFIG_JSON = {
    "architectures": ["DFlash2DraftModel"],
    "base_model": "moonshotai/Kimi-K3.1-Pro",
    "block_size": 5,
    "dflash_config": {
        "attention_mode": "mla",
        "aux_hidden_stream": "attn_res",
        "block_size": 5,
        "conv_group_size": 16,
        "conv_kernel_size": 2,
        "mask_token_id": 163824,
        "selector_rank": 256,
        "selector_top_k": 32,
        "target_layer_ids": [7, 23, 51, 67, 83, 91],
    },
    "head_dim": 64,
    "hidden_size": 7168,
    "kv_lora_rank": 512,
    "layer_types": ["sliding_attention"] * 5,
    "max_position_embeddings": 1048576,
    "mla_use_nope": False,
    "model_type": "qwen3",
    "num_attention_heads": 96,
    "num_hidden_layers": 5,
    "num_key_value_heads": 96,
    "num_target_layers": 93,
    "q_lora_rank": 1536,
    "qk_head_dim": 192,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "rms_norm_eps": 1e-05,
    "rope_interleave": True,
    "rope_parameters": {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 32.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 32768,
        "rope_theta": 50000.0,
        "rope_type": "yarn",
    },
    "rope_scaling": {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 32.0,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 32768,
        "rope_type": "yarn",
    },
    "rope_theta": 50000.0,
    "sliding_window": 1024,
    "use_sliding_window": True,
    "v_head_dim": 128,
}
K31_DFLASH2_CHECKPOINT = "/mnt/weka/model-cache/scratch/aling/moonshotai/Kimi-K3.1-Dflash2-Preview"


def _assert_k31_dflash2(cfg: DFlashDrafterConfig):
    assert cfg == kimi_k31_dflash2_mla_config()
    assert cfg.is_mla and not cfg.is_dspark
    # Six taps feed five layers: the fc is 6 x 7168 wide, and layer i's cache row does not depend on the count.
    assert cfg.target_layer_ids == (7, 23, 51, 67, 83, 91) and cfg.num_hidden_layers == 5
    assert cfg.target_feature_size == 43008
    assert (cfg.num_target_layers, cfg.hidden_size) == (93, 7168)  # the runtime's verifier cross-check
    assert (cfg.kv_lora_rank, cfg.qk_rope_head_dim, cfg.latent_dim, cfg.rope_dim) == (512, 64, 576, 64)
    assert cfg.head_dim == 128  # v_head_dim, not the config's Qwen3 `head_dim` of 64
    assert (cfg.rope_theta, cfg.rope_factor, cfg.rope_orig_max_pos) == (50000.0, 32.0, 32768)
    assert (cfg.rope_beta_fast, cfg.rope_beta_slow, cfg.rope_yarn_amplitude) == (32.0, 1.0, 1.0)
    assert cfg.rope_convention == "interleaved"  # config.json rope_interleave: true
    assert (cfg.block_size, cfg.mask_token_id, cfg.rms_norm_eps) == (5, 163824, 1e-5)


def test_k31_dflash2_config_json_parses_to_the_preset():
    _assert_k31_dflash2(DFlashDrafterConfig.from_hf_config(SimpleNamespace(**K31_DFLASH2_CONFIG_JSON)))
    # transformers 5 moves rope_theta into the rope dicts and drops the top-level key.
    moved = {k: v for k, v in K31_DFLASH2_CONFIG_JSON.items() if k != "rope_theta"}
    moved["rope_scaling"] = dict(moved["rope_parameters"])
    _assert_k31_dflash2(DFlashDrafterConfig.from_hf_config(SimpleNamespace(**moved)))


def test_k31_taps_fire_on_the_ranks_that_hold_them():
    """Tap t fires at layer t + 1's attention input (TtKimiK3Runtime._dflash_tap_layer), so on the four-galaxy
    split every rank owns at least one fc slice and the tail owns the last (91, at layer 92, the last layer)."""
    from models.demos.deepseek_v3_d_p.tt.kimi_k3.runtime import TtKimiK3Runtime

    cfg = kimi_k31_dflash2_mla_config()
    tap_layer = lambda t: TtKimiK3Runtime._dflash_tap_layer(None, t)  # noqa: E731
    splits = [(0, 24), (24, 24), (48, 24), (72, 21)]
    owned = [tuple(t for t in cfg.target_layer_ids if first <= tap_layer(t) < first + n) for first, n in splits]
    assert owned == [(7,), (23,), (51, 67), (83, 91)]
    assert max(tap_layer(t) for t in cfg.target_layer_ids) == cfg.num_target_layers - 1


def test_k31_reference_latent_takes_six_taps_into_five_layers():
    """The FC decomposes over taps (what lets it accumulate as the verifier streams, and ride the pipeline
    handoff as a partial sum), whatever the tap count; the result is one 576-wide row per token per layer."""
    cfg = kimi_k31_dflash2_mla_config(hidden_size=128)
    sd = synthetic_prefill_state_dict(cfg)
    assert sd["fc.weight"].shape == (128, 6 * 128)
    taps = [torch.randn(64, cfg.hidden_size) for _ in cfg.target_layer_ids]
    latents = context_latents(sd, taps, cfg)
    assert latents.shape == (5, 64, 576)
    fc = sd["fc.weight"].float()
    summed = sum(t @ fc[:, i * 128 : (i + 1) * 128].T for i, t in enumerate(taps))
    torch.testing.assert_close(summed, torch.cat(taps, dim=-1) @ fc.T, atol=1e-4, rtol=0)
    # Row order: [kv_a_layernorm(c_kv) 512 | rope(k_pe) 64]. Position 0 leaves k_pe unrotated.
    x = (summed * torch.rsqrt(summed.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)) * sd[
        "hidden_norm.weight"
    ].float()
    kv = x @ sd["layers.2.self_attn.kv_a_proj_with_mqa.weight"].float().T
    torch.testing.assert_close(latents[2, 0, 512:], kv[0, 512:], atol=1e-4, rtol=0)
    c_kv = kv[:, :512]
    normed = c_kv * torch.rsqrt(c_kv.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)
    torch.testing.assert_close(
        latents[2, :, :512], normed * sd["layers.2.self_attn.kv_a_layernorm.weight"].float(), atol=1e-4, rtol=0
    )


def test_drafter_context_window(tmp_path):
    assert drafter_context_window(str(tmp_path)) is None  # no config.json
    (tmp_path / "config.json").write_text(json.dumps(K31_DFLASH2_CONFIG_JSON))
    assert drafter_context_window(str(tmp_path)) == 1024
    for change in ({"use_sliding_window": False}, {"layer_types": ["sliding_attention"] * 4 + ["full_attention"]}):
        (tmp_path / "config.json").write_text(json.dumps({**K31_DFLASH2_CONFIG_JSON, **change}))
        assert drafter_context_window(str(tmp_path)) is None


def _k31_checkpoint():
    path = os.environ.get("K31_DFLASH2_HF_MODEL") or K31_DFLASH2_CHECKPOINT
    if not os.path.exists(os.path.join(path, "model.safetensors")):
        pytest.skip(f"no Kimi-K3.1 DFlash2 checkpoint at {path} (set K31_DFLASH2_HF_MODEL)")
    return path


def test_k31_real_checkpoint_config():
    """The checkpoint's own config.json, through the path the runtime takes (AutoConfig)."""
    path = _k31_checkpoint()
    with open(os.path.join(path, "config.json")) as f:
        on_disk = json.load(f)
    assert {k: on_disk.get(k) for k in K31_DFLASH2_CONFIG_JSON} == K31_DFLASH2_CONFIG_JSON
    _assert_k31_dflash2(DFlashDrafterConfig.from_pretrained(path))
    assert drafter_context_window(path) == 1024


def test_k31_real_checkpoint_keys_and_prefill_subset():
    """Key names and shapes of the real weights: `hidden_norm` (not `context_norm`), a 6-tap fc, and per layer
    `kv_a_proj_with_mqa` [576, 7168] and `kv_a_layernorm` [512]. Header first (cheap), then the loader itself:
    the prefill subset is ~0.7 GB of the 5.6 GB file."""
    path = _k31_checkpoint()
    cfg = kimi_k31_dflash2_mla_config()
    with open(os.path.join(path, "model.safetensors"), "rb") as f:
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    shapes = {k: tuple(v["shape"]) for k, v in header.items() if k != "__metadata__"}
    assert "context_norm.weight" not in shapes
    expected = {"fc.weight": (7168, 6 * 7168), "hidden_norm.weight": (7168,)}
    for i in range(cfg.num_hidden_layers):
        expected[f"layers.{i}.self_attn.kv_a_proj_with_mqa.weight"] = (576, 7168)
        expected[f"layers.{i}.self_attn.kv_a_layernorm.weight"] = (512,)
    assert {k: shapes.get(k) for k in expected} == expected
    assert not any(k.startswith(f"layers.{cfg.num_hidden_layers}.") for k in shapes)  # exactly five layers
    assert {header[k]["dtype"] for k in expected} == {"BF16"}

    tail = load_drafter_state_dict(path, build_kv_tail=True, attention_mode="mla")
    assert {k: tuple(v.shape) for k, v in tail.items()} == expected
    assert all(torch.isfinite(v.float()).all() for v in tail.values())
    # The reference runs on the real weights: one finite 576-wide row per token per layer.
    taps = [torch.randn(32, cfg.hidden_size) for _ in cfg.target_layer_ids]
    latents = context_latents(tail, taps, cfg, start=5088)
    assert latents.shape == (5, 32, 576) and torch.isfinite(latents).all()
    assert set(load_drafter_state_dict(path, build_kv_tail=False, attention_mode="mla")) == {"fc.weight"}
