# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: every tt_dit DiT attention module selects an explicit named recipe.

Each attention module names its recipe as ``sdpa_precision_default``; ``sdpa_precision=None`` resolves
to it and an explicit recipe overrides it. Constructors resolve the recipe before touching the mesh
device, so each is run with ``mesh_device=None`` until it fails on the absent device.
"""

import pytest

import ttnn
from models.tt_dit.blocks import attention as flux1_attention
from models.tt_dit.blocks import attention_opt as flux2_attention
from models.tt_dit.models.transformers import attention_mochi, attention_sd35, transformer_ideogram4
from models.tt_dit.models.transformers.ltx import attention_ltx
from models.tt_dit.models.transformers.minimax_h3 import attention_minimax_h3
from models.tt_dit.models.transformers.wan2_2 import attention_wan
from models.tt_dit.models.vae import vae, vae_sd35, vae_wan2_1
from models.tt_dit.models.vae.minimax_h3 import decoder_minimax_h3

BALANCED, STANDARD, FAST = ttnn.SDPAPrecision.BALANCED, ttnn.SDPAPrecision.STANDARD, ttnn.SDPAPrecision.FAST

# (class, constructor kwargs that reach the recipe resolution)
MODULES = {
    "flux1": (
        flux1_attention.Attention,
        dict(query_dim=512, head_dim=128, heads=4, out_dim=512, added_kv_proj_dim=512, eps=1e-6, padding_config=None),
    ),
    "flux2": (
        flux2_attention.Attention,
        dict(query_dim=512, head_dim=128, heads=4, out_dim=512, added_kv_proj_dim=512, eps=1e-6, padding_config=None),
    ),
    "sd35": (attention_sd35.SD35JointAttention, dict(query_dim=256, head_dim=64, heads=4)),
    "mochi": (
        attention_mochi.MochiAttention,
        dict(query_dim=512, added_kv_proj_dim=256, heads=4, head_dim=128),
    ),
    "wan": (attention_wan.WanAttention, dict(dim=512, num_heads=4)),
    "ltx": (attention_ltx.LTXAttention, dict(dim=512, num_heads=4)),
    "ideogram4": (
        transformer_ideogram4.Ideogram4TransformerBlock,
        dict(hidden_size=1024, intermediate_size=2048, num_heads=4, norm_eps=1e-6, adaln_dim=256),
    ),
    "minimax_h3": (
        attention_minimax_h3.MiniMaxH3Attention,
        dict(hidden_size=512, num_heads=4, head_dim=128),
    ),
}
DEVICE_ARGS = dict(mesh_device=None, ccl_manager=None, parallel_config=None)


def _resolve(name, **overrides):
    cls, kwargs = MODULES[name]
    obj = object.__new__(cls)
    try:
        cls.__init__(obj, **kwargs, **DEVICE_ARGS, **overrides)
    except (AttributeError, TypeError, AssertionError):
        pass  # the recipe is resolved before the (absent) mesh device is used
    return obj.sdpa_precision, obj.sdpa_kv_dtype


# Every denoiser attention defaults to STANDARD with BF16 K/V.
DEFAULTS = {name: (STANDARD, ttnn.bfloat16) for name in MODULES}


@pytest.mark.parametrize("name", sorted(MODULES))
def test_denoiser_default_recipe(name):
    assert MODULES[name][0].sdpa_precision_default == DEFAULTS[name][0]


def test_vae_default_recipes():
    assert vae_wan2_1.WanAttentionBlock.sdpa_precision_default == BALANCED
    assert decoder_minimax_h3.MiniMaxH3ViTAttention.sdpa_precision_default == BALANCED
    assert vae.VaeAttention.sdpa_precision_default == BALANCED
    assert vae_sd35.Attention.sdpa_precision_default == BALANCED


@pytest.mark.parametrize("name", sorted(MODULES))
def test_none_selects_the_default(name):
    assert _resolve(name) == DEFAULTS[name]


@pytest.mark.parametrize("name", sorted(MODULES))
def test_explicit_recipe_overrides_the_default(name):
    assert _resolve(name, sdpa_precision=BALANCED) == (BALANCED, ttnn.bfloat16)
    assert _resolve(name, sdpa_precision=FAST, sdpa_kv_dtype=ttnn.bfloat8_b) == (
        FAST,
        ttnn.bfloat8_b,
    )


@pytest.mark.parametrize("name", sorted(MODULES))
def test_low_precision_kv_needs_fast(expect_error, name):
    # A STANDARD default keeps BF16 KV; a packed KV dtype needs an explicit FAST.
    with expect_error(ValueError, "requires the FAST recipe"):
        _resolve(name, sdpa_kv_dtype=ttnn.bfloat8_b)
