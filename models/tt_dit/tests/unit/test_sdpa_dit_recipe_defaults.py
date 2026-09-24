# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: every tt_dit SDPA call selects an explicit named recipe on Blackhole.

Each attention module names its recipe as ``sdpa_precision_default``; ``sdpa_precision=None`` resolves
to it on Blackhole, an explicit recipe overrides it, and off Blackhole the module keeps its legacy SDPA
configuration (recipes are Blackhole-only). Constructors resolve the recipe before touching the mesh
device, so each is run with ``mesh_device=None`` until it fails on the absent device.
"""

import pytest
import ttnn

from models.tt_dit.blocks import attention as flux1_attention
from models.tt_dit.blocks import attention_opt as flux2_attention
from models.tt_dit.models.transformers import attention_mochi, attention_sd35, transformer_ideogram4
from models.tt_dit.models.transformers.ltx import attention_ltx
from models.tt_dit.models.transformers.ltx.quant_config import LtxQuantProfile
from models.tt_dit.models.transformers.minimax_h3 import attention_minimax_h3
from models.tt_dit.models.transformers.wan2_2 import attention_wan
from models.tt_dit.models.vae import vae_wan2_1
from models.tt_dit.models.vae.minimax_h3 import decoder_minimax_h3
from models.tt_dit.pipelines.wan.quant_config import QuantConfig

BALANCED, FAST, LOW = ttnn.SDPAPrecision.BALANCED, ttnn.SDPAPrecision.FAST, ttnn.SDPAPrecision.LOW_PRECISION

# (module, class, constructor kwargs that reach the recipe resolution)
MODULES = {
    "flux1": (
        flux1_attention,
        flux1_attention.Attention,
        dict(query_dim=512, head_dim=128, heads=4, out_dim=512, added_kv_proj_dim=512, eps=1e-6, padding_config=None),
    ),
    "flux2": (
        flux2_attention,
        flux2_attention.Attention,
        dict(query_dim=512, head_dim=128, heads=4, out_dim=512, added_kv_proj_dim=512, eps=1e-6, padding_config=None),
    ),
    "sd35": (attention_sd35, attention_sd35.SD35JointAttention, dict(query_dim=256, head_dim=64, heads=4)),
    "mochi": (
        attention_mochi,
        attention_mochi.MochiAttention,
        dict(query_dim=512, added_kv_proj_dim=256, heads=4, head_dim=128),
    ),
    "wan": (attention_wan, attention_wan.WanAttention, dict(dim=512, num_heads=4)),
    "ltx": (attention_ltx, attention_ltx.LTXAttention, dict(dim=512, num_heads=4)),
    "ideogram4": (
        transformer_ideogram4,
        transformer_ideogram4.Ideogram4TransformerBlock,
        dict(hidden_size=1024, intermediate_size=2048, num_heads=4, norm_eps=1e-6, adaln_dim=256),
    ),
    "minimax_h3": (
        attention_minimax_h3,
        attention_minimax_h3.MiniMaxH3Attention,
        dict(hidden_size=512, num_heads=4, head_dim=128),
    ),
}
DEVICE_ARGS = dict(mesh_device=None, ccl_manager=None, parallel_config=None)


def _resolve(monkeypatch, name, blackhole, **overrides):
    module, cls, kwargs = MODULES[name]
    monkeypatch.setattr(module, "is_blackhole", lambda: blackhole)
    obj = object.__new__(cls)
    try:
        cls.__init__(obj, **kwargs, **DEVICE_ARGS, **overrides)
    except (AttributeError, TypeError, AssertionError):
        pass  # the recipe is resolved before the (absent) mesh device is used
    return obj.sdpa_precision, obj.sdpa_kv_dtype


@pytest.mark.parametrize("name", sorted(MODULES))
def test_denoiser_default_recipe_is_balanced(name):
    assert MODULES[name][1].sdpa_precision_default == BALANCED


def test_vae_default_recipes():
    assert vae_wan2_1.WanAttentionBlock.sdpa_precision_default == BALANCED
    assert decoder_minimax_h3.MiniMaxH3ViTAttention.sdpa_precision_default == BALANCED


@pytest.mark.parametrize("name", sorted(MODULES))
def test_none_selects_the_default_on_blackhole(monkeypatch, name):
    assert _resolve(monkeypatch, name, True) == (BALANCED, ttnn.bfloat16)


@pytest.mark.parametrize("name", sorted(MODULES))
def test_explicit_recipe_overrides_the_default(monkeypatch, name):
    assert _resolve(monkeypatch, name, True, sdpa_precision=FAST) == (FAST, ttnn.bfloat16)
    assert _resolve(monkeypatch, name, True, sdpa_precision=LOW, sdpa_kv_dtype=ttnn.bfloat8_b) == (
        LOW,
        ttnn.bfloat8_b,
    )


@pytest.mark.parametrize("name", sorted(MODULES))
def test_legacy_off_blackhole(monkeypatch, name):
    assert _resolve(monkeypatch, name, False) == (None, ttnn.bfloat16)
    with pytest.raises(ValueError):
        _resolve(monkeypatch, name, False, sdpa_precision=FAST)


@pytest.mark.parametrize("name", sorted(MODULES))
def test_low_precision_kv_needs_low_precision(monkeypatch, name):
    # The default (BALANCED) keeps BF16 KV; a packed KV dtype needs an explicit LOW_PRECISION.
    with pytest.raises(ValueError):
        _resolve(monkeypatch, name, True, sdpa_kv_dtype=ttnn.bfloat8_b)


def test_ltx_quant_profile_self_attention_recipe(monkeypatch):
    # The shipped BFP8-input legacy tier maps to FAST on BF16 inputs (self-attention only).
    from models.tt_dit.models.transformers.ltx import quant_config

    monkeypatch.setattr(quant_config, "LTX_QUANT_ACTIVATIONS", True)
    profile = LtxQuantProfile.all_bf8_lofi()
    assert profile.sdpa_self_recipe() == (FAST, None)
    assert _resolve(monkeypatch, "ltx", True, quant_config=profile, is_self=True) == (FAST, ttnn.bfloat16)
    assert _resolve(monkeypatch, "ltx", True, quant_config=profile, is_self=False) == (BALANCED, ttnn.bfloat16)
    monkeypatch.setattr(quant_config, "LTX_QUANT_ACTIVATIONS", False)
    assert profile.sdpa_self_recipe() == (None, None)


def test_wan_quant_presets_map_to_recipes():
    assert QuantConfig.default().ring_sdpa.precision is None  # keeps WanAttention's default
    assert QuantConfig.all_lofi().ring_sdpa.precision == FAST
    assert QuantConfig.all_bf8_lofi().ring_sdpa.precision == FAST
