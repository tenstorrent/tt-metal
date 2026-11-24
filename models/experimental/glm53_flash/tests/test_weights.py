# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``tt/weights.py``: the HF -> tile-cache layout conversion (checked against the HF modules' own math) and
``TorchStateDictSource``'s device tensors (dtype map, layout, values)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.experimental.glm53_flash.tests.test_kda_block import init_like_checkpoint, pcc
from models.experimental.glm53_flash.tt.config import Glm5NextTextConfig as TtConfig
from models.experimental.glm53_flash.tt.precision import WEIGHT_DTYPES
from models.experimental.glm53_flash.tt.weights import (
    TileCacheSource,
    TorchStateDictSource,
    cache_dtype,
    reference_to_cache_layout,
)

transformers_glm = pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig as HfConfig  # noqa: E402

pytestmark = pytest.mark.use_module_device({"l1_small_size": 24576})

KDA_DENSE, MLA_MOE = 0, 1
SMALL = dict(
    hidden_size=256,
    intermediate_size=512,
    moe_intermediate_size=128,
    n_routed_experts=8,
    num_experts_per_tok=2,
    linear_num_heads=4,
    linear_head_dim=64,
    num_attention_heads=4,
    num_key_value_heads=4,
    q_lora_rank=64,
    kv_lora_rank=64,
    qk_nope_head_dim=32,
    v_head_dim=48,
    index_n_heads=4,
    index_head_dim=32,
    num_hidden_layers=2,
    layer_types=["linear_attention", "indexed_attention"],
    mlp_layer_types=["dense", "sparse"],
)


def _hf_config():
    kw = dict(SMALL)
    kw["linear_attn_config"] = {"head_dim": kw.pop("linear_head_dim"), "num_heads": kw.pop("linear_num_heads")}
    return HfConfig(**kw)


@pytest.fixture(scope="module")
def cfg():
    return TtConfig(**SMALL)


@pytest.fixture(scope="module", params=[KDA_DENSE, MLA_MOE], ids=["kda_dense", "mla_moe"])
def layer(request):
    return request.param, init_like_checkpoint(
        transformers_glm.Glm5NextTextDecoderLayer(_hf_config(), request.param), seed=1
    )


# ---------------------------------------------------------------------------------------------
# reference_to_cache_layout
# ---------------------------------------------------------------------------------------------


def test_every_reference_tensor_is_converted(layer, cfg):
    """Every HF tensor lands somewhere (by element count) and nothing appears from nowhere."""
    _, ref = layer
    sd = ref.state_dict()
    out = reference_to_cache_layout(sd, cfg)
    assert sum(t.numel() for t in out.values()) == sum(t.numel() for t in sd.values())
    assert not any(k.startswith(("attn_hc.", "ffn_hc.", "self_attn.forget_gate.")) for k in out)
    for which in ("attn", "ffn"):
        assert out[f"hc_{which}_fn"].shape == sd[f"{which}_hc.fn"].t().shape


def test_linear_weights_become_in_by_out(layer, cfg):
    """x @ cache[name] == HF Linear(x) for every plain 2-D projection."""
    _, ref = layer
    out = reference_to_cache_layout(ref.state_dict(), cfg)
    mods = {n: m for n, m in ref.named_modules() if isinstance(m, torch.nn.Linear)}
    checked = 0
    for name, mod in mods.items():
        key = f"{name}.weight"
        if key == "self_attn.kv_b_proj.weight" or name.startswith(("attn_hc", "ffn_hc")):
            continue
        if name.startswith("self_attn.forget_gate."):
            key = "self_attn." + key[len("self_attn.forget_gate.") :]
        x = torch.randn(3, mod.in_features)
        torch.testing.assert_close(x @ out[key], F.linear(x, mod.weight), rtol=1e-5, atol=1e-5)
        checked += 1
    assert checked >= 6


def test_kda_conv_taps(cfg):
    """The fused depthwise conv [3C, 1, 4] -> q/k/v taps [4, C]: causal conv with tap j weighting x[t - 3 + j]."""
    ref = init_like_checkpoint(transformers_glm.Glm5NextTextDecoderLayer(_hf_config(), KDA_DENSE), seed=2)
    w = ref.self_attn.conv1d.weight  # [3C, 1, K]
    out = reference_to_cache_layout(ref.state_dict(), cfg)
    c = cfg.linear_num_heads * cfg.linear_head_dim
    taps = torch.cat([out[f"self_attn.{p}_conv1d.weight"] for p in "qkv"], dim=1)
    assert taps.shape == (4, 3 * c)
    x = torch.randn(1, 3 * c, 10)
    want = F.conv1d(F.pad(x, (3, 0)), w, groups=3 * c)
    xp = F.pad(x, (3, 0))
    got = sum(taps[j][None, :, None] * xp[:, :, j : j + 10] for j in range(4))
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_mla_kv_b_split(cfg):
    """W_uk [H, nope, kv_lora] and W_uv [H, kv_lora, v] reproduce kv_b_proj's per-head k_nope and value."""
    ref = init_like_checkpoint(transformers_glm.Glm5NextTextDecoderLayer(_hf_config(), MLA_MOE), seed=3)
    out = reference_to_cache_layout(ref.state_dict(), cfg)
    h, nope, v, r = cfg.num_attention_heads, cfg.qk_nope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank
    assert out["self_attn.kv_b_proj.W_uk"].shape == (h, nope, r)
    assert out["self_attn.kv_b_proj.W_uv"].shape == (h, r, v)
    lat = torch.randn(5, r)
    kv = ref.self_attn.kv_b_proj(lat).view(5, h, nope + v)
    k_nope = torch.einsum("tr,hnr->thn", lat, out["self_attn.kv_b_proj.W_uk"])
    value = torch.einsum("tr,hrv->thv", lat, out["self_attn.kv_b_proj.W_uv"])
    torch.testing.assert_close(k_nope, kv[..., :nope], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(value, kv[..., nope:], rtol=1e-5, atol=1e-5)


def test_routed_experts_layout(cfg):
    ref = init_like_checkpoint(transformers_glm.Glm5NextTextDecoderLayer(_hf_config(), MLA_MOE), seed=4)
    out = reference_to_cache_layout(ref.state_dict(), cfg)
    e, i, d = cfg.n_routed_experts, cfg.moe_intermediate_size, cfg.hidden_size
    gu, dn = out["mlp.experts.gate_up_proj"], out["mlp.experts.down_proj"]
    assert gu.shape == (e, d, 2 * i) and dn.shape == (e, i, d)
    x = torch.randn(3, d)
    for ex in (0, e - 1):
        torch.testing.assert_close(x @ gu[ex], F.linear(x, ref.mlp.experts.gate_up_proj[ex]), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------------------------
# dtype map + TorchStateDictSource on device
# ---------------------------------------------------------------------------------------------


def test_cache_dtype_map_covers_the_reference(layer, cfg):
    _, ref = layer
    out = reference_to_cache_layout(ref.state_dict(), cfg)
    for name in out:
        dt = cache_dtype(name)
        if ".experts." in name:
            assert dt == WEIGHT_DTYPES["routed_expert"]
        elif name.endswith(("A_log", "dt_bias", "e_score_correction_bias")) or name in (
            "hc_attn_base",
            "hc_attn_scale",
            "hc_ffn_base",
            "hc_ffn_scale",
        ):
            assert dt == ttnn.float32, name


def test_state_dict_source_device_tensors(device, layer, cfg, expect_error):
    idx, ref = layer
    src = TorchStateDictSource({idx: ref.state_dict()}, cfg)
    host = src.layers[idx]
    for name, t in host.items():
        if name.startswith("mlp.experts."):
            continue
        dt = cache_dtype(name)
        d = src.device(idx, name, device)
        assert d.dtype == dt and d.layout == ttnn.TILE_LAYOUT, name
        back = ttnn.to_torch(d).float().reshape(t.shape)
        if dt in (ttnn.float32, ttnn.bfloat16):
            torch.testing.assert_close(
                back, t.float().to(torch.bfloat16 if dt == ttnn.bfloat16 else torch.float32).float(), rtol=0, atol=0
            )
        else:
            assert pcc(back, t) > 0.999, name
        assert torch.equal(src.host(idx, name), t)
    with expect_error(KeyError, "missing"):
        src.device(idx, "self_attn.not_a_weight", device)


def test_state_dict_source_experts(device, cfg, expect_error):
    ref = init_like_checkpoint(transformers_glm.Glm5NextTextDecoderLayer(_hf_config(), MLA_MOE), seed=5)
    src = TorchStateDictSource({MLA_MOE: ref.state_dict()}, cfg)
    e = cfg.n_routed_experts
    gu = src.experts(MLA_MOE, "gate_up_proj", e, device)
    assert gu.dtype == WEIGHT_DTYPES["routed_expert"]
    assert tuple(gu.shape) == (1, e, cfg.hidden_size, 2 * cfg.moe_intermediate_size)
    back = ttnn.to_torch(gu).float()[0]
    assert pcc(back, src.layers[MLA_MOE]["mlp.experts.gate_up_proj"]) > 0.98  # bfloat4_b storage
    with expect_error(AssertionError, ""):
        src.experts(MLA_MOE, "gate_up_proj", e + 1, device)


def test_tile_cache_source_requires_files(tmp_path, expect_error):
    with expect_error(FileNotFoundError, "weight cache dir"):
        TileCacheSource(str(tmp_path / "missing"), str(tmp_path / "manifest.json"))
    with expect_error(FileNotFoundError, "manifest"):
        TileCacheSource(str(tmp_path), str(tmp_path / "manifest.json"))
