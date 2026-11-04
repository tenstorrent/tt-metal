# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Named SDPA precision recipes for tt_dit SDPA calls.

Every DiT denoiser attention in ``models/tt_dit`` selects a recipe explicitly
(tech_reports/FlashAttention/SDPAPrecisionRecipes.md); text/vision encoders and the H3 audio VAE encoder do not.
Each attention module names the recipe of each of its calls as a class-level default
(e.g. ``WanAttention.sdpa_precision_default``) and takes
``sdpa_precision: ttnn.SDPAPrecision | None`` / ``sdpa_kv_dtype: ttnn.DataType | None`` overrides:
``None`` selects the module's default recipe, anything else replaces it for every call of the module
(``FAST`` with ``sdpa_kv_dtype`` for low-precision KV). ``resolve_recipe`` does this.

DiT denoisers default to STANDARD (FP32 reference-max state, fused K chunks): it is faster per core than the
legacy kernel at every chunk size (1.07x geomean, D128) and far more accurate at long K. Pass ``sdpa_precision``
to opt up (e.g. BALANCED) or down (FAST).

Recipe blocking is op-selected: a recipe config
carries only the caller's grid and leaves ``q_chunk_size``/``k_chunk_size`` at 0, and SDPA chooses the
chunks (and, for exp ring, the grid width) from the shape, recipe, op, grid and L1.
"""

from __future__ import annotations

import ttnn

RECIPE_KV_DTYPES = (ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b)


def resolve_recipe(
    precision: ttnn.SDPAPrecision | None,
    kv_dtype: ttnn.DataType | None,
    *,
    default: ttnn.SDPAPrecision,
    head_dim: int,
    model: str,
) -> tuple[ttnn.SDPAPrecision, ttnn.DataType]:
    """The validated ``(recipe, KV storage dtype)`` a module runs: the caller's ``precision`` override, else the
    module's ``default``; KV is BF16 unless FAST asks otherwise."""
    precision = default if precision is None else precision
    return precision, validate_recipe_args(precision, kv_dtype, head_dim=head_dim, model=model)


def recipe_config(grid) -> ttnn.SDPAProgramConfig:
    """A recipe program config: the grid only; SDPA chooses the chunks (exp_approx_mode is the recipe's)."""
    grid = grid if isinstance(grid, ttnn.CoreCoord) else ttnn.CoreCoord(*grid)
    return ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid)


def validate_recipe_args(
    precision: ttnn.SDPAPrecision,
    kv_dtype: ttnn.DataType | None,
    *,
    head_dim: int,
    model: str,
) -> ttnn.DataType:
    """Validate a recipe and return its KV storage dtype (BF16 unless FAST asks otherwise)."""
    kv_dtype = kv_dtype or ttnn.bfloat16
    if head_dim <= 0 or head_dim % ttnn.TILE_SIZE != 0:
        raise ValueError(f"{model}: named SDPA recipes require a tile-aligned head_dim (head_dim={head_dim})")
    if kv_dtype not in RECIPE_KV_DTYPES:
        raise ValueError(f"{model}: unsupported SDPA KV dtype {kv_dtype}")
    if kv_dtype != ttnn.bfloat16 and precision != ttnn.SDPAPrecision.FAST:
        raise ValueError(f"{model}: low-precision KV requires the FAST recipe")
    return kv_dtype


def prepare_recipe_inputs(precision: ttnn.SDPAPrecision, kv_dtype: ttnn.DataType, q, k, v):
    """FAST input preparation (after norm/RoPE, before any ring communication).

    Returns the inputs unchanged for every other recipe.
    """
    if precision != ttnn.SDPAPrecision.FAST:
        return q, k, v
    prepare = ttnn.transformer.prepare_sdpa_input
    return (
        prepare(q, is_query=True),
        prepare(k, is_query=False, dtype=kv_dtype),
        prepare(v, is_query=False, dtype=kv_dtype),
    )
