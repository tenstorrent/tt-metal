# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: shared tt_dit SDPA recipe opt-in helpers (no device)."""

import pytest

import ttnn
from models.tt_dit.utils import sdpa_recipe as recipe

STANDARD, FAST = ttnn.SDPAPrecision.STANDARD, ttnn.SDPAPrecision.FAST


def test_validate_recipe():
    for head_dim in (64, 128, 256):
        assert recipe.validate_recipe_args(STANDARD, None, head_dim=head_dim, model="m") == ttnn.bfloat16
    assert recipe.validate_recipe_args(FAST, ttnn.bfloat4_b, head_dim=128, model="m") == ttnn.bfloat4_b


@pytest.mark.parametrize(
    "precision, kv_dtype, head_dim",
    [
        (STANDARD, None, 96),  # D96 is not a recipe head dim
        (STANDARD, None, 512),
        (STANDARD, ttnn.bfloat8_b, 128),  # packed KV needs FAST
        (FAST, ttnn.float32, 128),  # unsupported storage
    ],
)
def test_validate_rejects(expect_error, precision, kv_dtype, head_dim):
    with expect_error(ValueError, ""):
        recipe.validate_recipe_args(precision, kv_dtype, head_dim=head_dim, model="m")


def test_resolve_recipe(expect_error):
    # None selects the module's default recipe; an explicit recipe overrides it. Both are validated.
    args = dict(default=STANDARD, head_dim=128, model="m")
    assert recipe.resolve_recipe(None, None, **args) == (STANDARD, ttnn.bfloat16)
    assert recipe.resolve_recipe(FAST, ttnn.bfloat8_b, **args) == (FAST, ttnn.bfloat8_b)
    with expect_error(ValueError, "FAST"):
        recipe.resolve_recipe(None, ttnn.bfloat8_b, **args)  # KV dtype without FAST


def test_recipe_config_is_grid_only():
    for grid in ((11, 9), ttnn.CoreCoord(10, 10)):
        config = recipe.recipe_config(grid)
        assert (config.compute_with_storage_grid_size.x, config.compute_with_storage_grid_size.y) == tuple(
            grid if isinstance(grid, tuple) else (grid.x, grid.y)
        )
        assert (config.q_chunk_size, config.k_chunk_size) == (0, 0)  # SDPA chooses the blocking
        assert config.exp_approx_mode is None and config.max_cores_per_head_batch is None


def test_prepare_passthrough():
    sentinel = object(), object(), object()
    assert recipe.prepare_recipe_inputs(STANDARD, ttnn.bfloat16, *sentinel) == sentinel
