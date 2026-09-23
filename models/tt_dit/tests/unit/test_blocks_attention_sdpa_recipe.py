# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: opt-in SDPA recipe wiring of the shared joint Attention block (FLUX.1, Qwen-Image, Motif).

The block's helpers are exercised on instances built with object.__new__, which bypasses __init__
(it needs a mesh device). Constructor validation runs before any device access.
"""

import pytest

import ttnn
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


def test_legacy_program_config_is_the_tuned_object():
    attention = _bare_attention(None, q_chunk=64, k_chunk=1024)
    assert attention._sdpa_program_config(ring=False) is attention.sdpa_program_config
    assert attention._sdpa_program_config(ring=True) is attention.sdpa_program_config


def test_legacy_sdpa_kwargs_follow_reassigned_compute_config():
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


def test_recipe_keeps_supported_tuned_chunks():
    attention = _bare_attention(ttnn.SDPAPrecision.ACCURATE, q_chunk=128, k_chunk=256)
    recipe = attention._sdpa_program_config(ring=True)
    assert recipe is not attention.sdpa_program_config
    assert _chunks(recipe) == (128, 256)


def test_recipe_falls_back_for_unsupported_chunks():
    # Motif-style K1024 and FLUX (BH, sp8, tp4) Q64 are outside the recipe blocking rules.
    attention = _bare_attention(ttnn.SDPAPrecision.ACCURATE, q_chunk=64, k_chunk=1024)
    assert _chunks(attention._sdpa_program_config(ring=False)) == (256, 512)
    assert _chunks(attention._sdpa_program_config(ring=True)) == (256, 512)


def test_ring_recipe_needs_even_q_tiles():
    attention = _bare_attention(ttnn.SDPAPrecision.ACCURATE, q_chunk=224, k_chunk=384)
    assert _chunks(attention._sdpa_program_config(ring=False)) == (224, 384)  # joint: odd tiles allowed
    assert _chunks(attention._sdpa_program_config(ring=True)) == (256, 384)  # ring: 7 tiles -> Q256


@pytest.mark.parametrize("key", sorted(Flux1Transformer.sdpa_chunk_size_map))
def test_flux_tuned_chunks_map_to_recipe_chunks(key):
    q_chunk, k_chunk = Flux1Transformer.sdpa_chunk_size_map[key]
    attention = _bare_attention(ttnn.SDPAPrecision.ACCURATE, q_chunk=q_chunk, k_chunk=k_chunk)
    q, k = _chunks(attention._sdpa_program_config(ring=True))
    assert q % 32 == 0 and 128 <= q <= 320 and (q // 32) % 2 == 0
    assert k in (256, 384, 512)
    assert q == (q_chunk if q_chunk >= 128 else 256) and k == k_chunk


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


def test_attention_rejects_recipe_for_d64():
    with pytest.raises(ValueError, match="D128"):
        _construct_attention(64, ttnn.SDPAPrecision.ACCURATE)


def test_attention_rejects_low_precision_kv_without_low_precision_recipe():
    with pytest.raises(ValueError):
        _construct_attention(128, None, ttnn.bfloat8_b)
    with pytest.raises(ValueError):
        _construct_attention(128, ttnn.SDPAPrecision.ACCURATE, ttnn.bfloat8_b)


@pytest.mark.parametrize("precision", [ttnn.SDPAPrecision.ACCURATE, ttnn.SDPAPrecision.LOW_PRECISION])
def test_motif_rejects_any_recipe(precision):
    assert MOTIF_6B_CONFIG.head_dim == 64
    with pytest.raises(ValueError, match="Motif"):
        MotifTransformer.validate_sdpa_recipe(MOTIF_6B_CONFIG, precision, None)
    with pytest.raises(ValueError, match="Motif"):
        MotifTransformer(
            config=MOTIF_6B_CONFIG,
            latents_height=128,
            latents_width=128,
            mesh_device=None,
            ccl_manager=None,
            parallel_config=None,
            sdpa_precision=precision,
        )
    MotifTransformer.validate_sdpa_recipe(MOTIF_6B_CONFIG, None, None)  # legacy path is untouched
