# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: SD3.5 (D64) opt-in SDPA recipe wiring of SD35JointAttention.

Instances are built with object.__new__ (no mesh device); only the program-config / kwargs helpers
and constructor validation (which runs before any device access) are exercised.
"""

import pytest

import ttnn
from models.tt_dit.models.transformers.attention_sd35 import SD35JointAttention
from models.tt_dit.models.transformers.transformer_sd35 import SD35Transformer2DModel

GRID = (12, 9)


def _bare_attention(precision, q_chunk=256, k_chunk=512):
    attention = object.__new__(SD35JointAttention)
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
    attention = _bare_attention(None)
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
        (ttnn.SDPAPrecision.FAST, False),
        (ttnn.SDPAPrecision.ACCURATE, False),
        (ttnn.SDPAPrecision.LOW_PRECISION, True),
    ],
)
def test_recipe_sdpa_kwargs_replace_compute_config(precision, prepared):
    assert _bare_attention(precision)._sdpa_kwargs() == {"precision": precision, "inputs_prepared": prepared}


@pytest.mark.parametrize(
    ("q_chunk", "k_chunk"),
    sorted({*SD35JointAttention.sdpa_chunk_size_map.values(), SD35JointAttention.default_sdpa_chunk_size, (224, 1024)}),
)
def test_recipe_chunks_are_op_selected(q_chunk, k_chunk):
    attention = _bare_attention(ttnn.SDPAPrecision.ACCURATE, q_chunk=q_chunk, k_chunk=k_chunk)
    for ring in (False, True):
        recipe = attention._sdpa_program_config(ring=ring)
        assert recipe is not attention.sdpa_program_config
        assert _chunks(recipe) == (0, 0)


@pytest.mark.parametrize(
    ("precision", "kv_dtype"),
    [
        (ttnn.SDPAPrecision.FAST, None),
        (ttnn.SDPAPrecision.ACCURATE, None),
        (ttnn.SDPAPrecision.LOW_PRECISION, ttnn.bfloat8_b),
    ],
)
def test_sd35_d64_recipes_accepted(precision, kv_dtype):
    SD35Transformer2DModel.validate_sdpa_recipe(None, None, head_dim=64)  # legacy: no-op
    SD35Transformer2DModel.validate_sdpa_recipe(precision, kv_dtype, head_dim=64)


def test_sd35_invalid_recipe_args_rejected():
    with pytest.raises(ValueError, match="head_dim"):
        SD35Transformer2DModel.validate_sdpa_recipe(ttnn.SDPAPrecision.ACCURATE, None, head_dim=96)
    with pytest.raises(ValueError, match="LOW_PRECISION"):
        SD35Transformer2DModel.validate_sdpa_recipe(ttnn.SDPAPrecision.ACCURATE, ttnn.bfloat8_b, head_dim=64)
    with pytest.raises(ValueError, match="LOW_PRECISION"):
        SD35Transformer2DModel.validate_sdpa_recipe(None, ttnn.bfloat8_b, head_dim=64)


def test_attention_constructor_validates_before_device_access():
    kwargs = dict(query_dim=384, heads=4, mesh_device=None, ccl_manager=None, parallel_config=None)
    with pytest.raises(ValueError, match="SD3.5"):
        SD35JointAttention(head_dim=96, sdpa_precision=ttnn.SDPAPrecision.ACCURATE, **kwargs)
    with pytest.raises(ValueError, match="SD3.5"):
        SD35JointAttention(head_dim=64, sdpa_kv_dtype=ttnn.bfloat8_b, **kwargs)
    # A valid D64 recipe passes validation and then reaches the (absent) parallel config / device.
    with pytest.raises(AttributeError):
        SD35JointAttention(head_dim=64, sdpa_precision=ttnn.SDPAPrecision.ACCURATE, **kwargs)
