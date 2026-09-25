# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: SDPA recipe wiring of the shared joint Attention block (FLUX.1, Qwen-Image, Motif).

The block's helpers are exercised on instances built with object.__new__, which bypasses __init__
(it needs a mesh device). Constructor validation runs before any device access. The default recipe
itself is covered by test_sdpa_dit_recipe_defaults.py.
"""

import pytest

import ttnn
from models.tt_dit.blocks import attention as attention_module
from models.tt_dit.blocks.attention import Attention
from models.tt_dit.models.transformers.transformer_flux1 import Flux1Transformer
from models.tt_dit.models.transformers.transformer_motif import MOTIF_6B_CONFIG, MotifTransformer

GRID = (12, 9)


def _bare_attention(precision, q_chunk=128, k_chunk=512):
    attention = object.__new__(Attention)
    attention.sdpa_precision = precision
    attention.sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=GRID,
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )
    attention.sdpa_compute_kernel_config = "legacy"
    return attention


def _chunks(program_config):
    return program_config.q_chunk_size, program_config.k_chunk_size


def test_legacy_sdpa_kwargs_follow_reassigned_compute_config():
    # Legacy (non-Blackhole) path.
    attention = _bare_attention(None)
    assert attention._sdpa_kwargs() == {"compute_kernel_config": "legacy"}
    attention.sdpa_compute_kernel_config = "reassigned"
    assert attention._sdpa_kwargs() == {"compute_kernel_config": "reassigned"}


@pytest.mark.parametrize(
    ("precision", "prepared"),
    [
        (ttnn.SDPAPrecision.ACCURATE, False),
        (ttnn.SDPAPrecision.COMPENSATED, False),
        (ttnn.SDPAPrecision.LOW_PRECISION, True),
    ],
)
def test_recipe_sdpa_kwargs_replace_compute_config(precision, prepared):
    attention = _bare_attention(precision)
    assert attention._sdpa_kwargs() == {"precision": precision, "inputs_prepared": prepared}


def test_flux_tuned_chunks_are_legacy_only():
    # On Blackhole every call runs a recipe with op-selected chunks; the table serves other archs only.
    assert Flux1Transformer.sdpa_chunk_size_map
    assert all(not blackhole for blackhole, _sp, _tp in Flux1Transformer.sdpa_chunk_size_map)


@pytest.fixture
def blackhole(monkeypatch):
    monkeypatch.setattr(attention_module, "is_blackhole", lambda: True)


def _construct_attention(head_dim, precision, kv_dtype=None):
    # Validation precedes any mesh-device access, so mesh_device=None is never touched on failure.
    return Attention(
        query_dim=head_dim * 4,
        head_dim=head_dim,
        heads=4,
        out_dim=head_dim * 4,
        added_kv_proj_dim=head_dim * 4,
        eps=1e-6,
        mesh_device=None,
        ccl_manager=None,
        parallel_config=None,
        padding_config=None,
        sdpa_precision=precision,
        sdpa_kv_dtype=kv_dtype,
    )


@pytest.mark.parametrize("precision", [None, ttnn.SDPAPrecision.ACCURATE])
def test_attention_rejects_recipe_for_unsupported_head_dim(blackhole, precision):
    # None selects the default recipe, so D96 is rejected either way on Blackhole.
    with pytest.raises(ValueError, match="head_dim"):
        _construct_attention(96, precision)


@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("precision", [None, ttnn.SDPAPrecision.ACCURATE])
def test_attention_accepts_recipe_head_dims(blackhole, head_dim, precision):
    # Validation passes; construction then reaches the (absent) mesh device.
    with pytest.raises(AttributeError):
        _construct_attention(head_dim, precision)


def test_attention_rejects_low_precision_kv_without_low_precision_recipe(blackhole):
    with pytest.raises(ValueError):
        _construct_attention(128, None, ttnn.bfloat8_b)
    with pytest.raises(ValueError):
        _construct_attention(128, ttnn.SDPAPrecision.ACCURATE, ttnn.bfloat8_b)


@pytest.mark.parametrize(
    ("precision", "kv_dtype"),
    [(ttnn.SDPAPrecision.ACCURATE, None), (ttnn.SDPAPrecision.LOW_PRECISION, ttnn.bfloat8_b)],
)
def test_motif_accepts_d64_recipes(precision, kv_dtype):
    assert MOTIF_6B_CONFIG.head_dim == 64
    MotifTransformer.validate_sdpa_recipe(MOTIF_6B_CONFIG, precision, kv_dtype)
    MotifTransformer.validate_sdpa_recipe(MOTIF_6B_CONFIG, None, None)  # the default recipe
    with pytest.raises(ValueError, match="LOW_PRECISION"):
        MotifTransformer.validate_sdpa_recipe(MOTIF_6B_CONFIG, ttnn.SDPAPrecision.ACCURATE, ttnn.bfloat8_b)
    with pytest.raises(ValueError, match="LOW_PRECISION"):
        MotifTransformer.validate_sdpa_recipe(MOTIF_6B_CONFIG, None, ttnn.bfloat8_b)
