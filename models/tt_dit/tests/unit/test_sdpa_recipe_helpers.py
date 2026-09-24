# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: shared tt_dit SDPA recipe opt-in helpers (no device)."""

import pytest
import ttnn

from models.tt_dit.utils import sdpa_recipe as recipe

FAST, COMPENSATED, LOW = ttnn.SDPAPrecision.FAST, ttnn.SDPAPrecision.COMPENSATED, ttnn.SDPAPrecision.LOW_PRECISION


def test_validate_legacy_and_recipe():
    assert recipe.validate_recipe_args(None, None, head_dim=64, model="m") == ttnn.bfloat16
    for head_dim in (64, 128, 256):
        assert recipe.validate_recipe_args(FAST, None, head_dim=head_dim, model="m") == ttnn.bfloat16
    assert recipe.exp_ring_supports(128) and not recipe.exp_ring_supports(64) and not recipe.exp_ring_supports(256)
    assert recipe.validate_recipe_args(LOW, ttnn.bfloat4_b, head_dim=128, model="m") == ttnn.bfloat4_b


@pytest.mark.parametrize(
    "precision, kv_dtype, head_dim, blackhole",
    [
        (FAST, None, 96, True),  # D96 is not a recipe head dim
        (FAST, None, 512, True),
        (FAST, None, 128, False),  # not Blackhole
        (COMPENSATED, ttnn.bfloat8_b, 128, True),  # packed KV needs LOW_PRECISION
        (LOW, ttnn.float32, 128, True),  # unsupported storage
        (None, ttnn.bfloat8_b, 128, True),  # KV dtype without a recipe
    ],
)
def test_validate_rejects(precision, kv_dtype, head_dim, blackhole):
    with pytest.raises(ValueError):
        recipe.validate_recipe_args(precision, kv_dtype, head_dim=head_dim, model="m", is_blackhole=blackhole)


def test_program_config_is_op_selected_and_keeps_grid():
    tuned = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(8, 4),
        q_chunk_size=352,
        k_chunk_size=128,
        exp_approx_mode=False,
        max_cores_per_head_batch=8,
    )
    for kwargs in ({}, {"ring": True}, {"exp_ring": True}):
        config = recipe.recipe_program_config(tuned, **kwargs)
        assert (config.compute_with_storage_grid_size.x, config.compute_with_storage_grid_size.y) == (8, 4)
        assert (config.q_chunk_size, config.k_chunk_size) == (0, 0)  # SDPA chooses the blocking
        assert config.exp_approx_mode is None and config.max_cores_per_head_batch == 8


def test_kwargs():
    assert recipe.sdpa_kwargs(None, "cfg") == {"compute_kernel_config": "cfg"}
    assert recipe.sdpa_kwargs(COMPENSATED, "cfg") == {"precision": COMPENSATED, "inputs_prepared": False}
    assert recipe.recipe_sdpa_kwargs(LOW) == {"precision": LOW, "inputs_prepared": True}


def test_prepare_passthrough_and_mask():
    sentinel = object(), object(), object()
    assert recipe.prepare_recipe_inputs(None, ttnn.bfloat16, *sentinel) == sentinel
    assert recipe.prepare_recipe_inputs(COMPENSATED, ttnn.bfloat16, *sentinel) == sentinel
    assert recipe.prepare_recipe_inputs(LOW, ttnn.bfloat16, None, None, None) == (None, None, None)
    recipe.reject_mask(None, "mask", model="m")
    recipe.reject_mask(FAST, None, model="m")
    with pytest.raises(ValueError):
        recipe.reject_mask(FAST, "mask", model="m")
