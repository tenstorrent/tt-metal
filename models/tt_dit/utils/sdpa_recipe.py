# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Named SDPA precision recipes for tt_dit SDPA calls.

Every SDPA-variant call in ``models/tt_dit`` selects a recipe explicitly (docs/sdpa_precision.md,
docs/sdpa_recipe_consolidation.md task 7). Each attention module names the recipe of each of its
calls as a class-level default (e.g. ``WanAttention.sdpa_precision_default``) and takes
``sdpa_precision: ttnn.SDPAPrecision | None`` / ``sdpa_kv_dtype: ttnn.DataType | None`` overrides:
``None`` selects the module's default recipe, anything else replaces it for every call of the module
(``LOW_PRECISION`` with ``sdpa_kv_dtype`` for low-precision KV). ``resolve_precision`` does this.

Recipes are qualified on Blackhole only, so on other architectures ``resolve_precision`` returns
``None`` and the module keeps its legacy SDPA program/compute configuration (explicit chunks,
``compute_kernel_config``); nothing else in this module is consulted then.

Recipe blocking is op-selected (docs/sdpa_precision.md, "Op-selected blocking"): a recipe config
carries only the caller's grid and leaves ``q_chunk_size``/``k_chunk_size`` at 0, and SDPA chooses the
chunks (and, for exp ring, the grid width) from the shape, recipe, op, grid and L1. Exp ring joint SDPA
recipes are D128 only.
"""

from __future__ import annotations

import ttnn

RECIPE_HEAD_DIMS = (64, 128, 256)
EXP_RING_HEAD_DIMS = (128,)
RECIPE_KV_DTYPES = (ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b)


def resolve_precision(
    precision: ttnn.SDPAPrecision | None,
    default: ttnn.SDPAPrecision,
    *,
    blackhole: bool,
    model: str,
) -> ttnn.SDPAPrecision | None:
    """The recipe an SDPA call runs: the caller's ``precision`` override, else the call's ``default``.

    Returns ``None`` (the module's legacy configuration) off Blackhole, where recipes are not
    qualified; an explicit override there is an error.
    """
    if not blackhole:
        if precision is not None:
            raise ValueError(f"{model}: named SDPA recipes require Blackhole")
        return None
    return default if precision is None else precision


def recipe_config(grid, max_cores_per_head_batch: int | None = None) -> ttnn.SDPAProgramConfig:
    """A recipe program config: the grid only; SDPA chooses the chunks (exp_approx_mode is the recipe's)."""
    grid = grid if isinstance(grid, ttnn.CoreCoord) else ttnn.CoreCoord(*grid)
    if max_cores_per_head_batch is None:
        return ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid)
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid, max_cores_per_head_batch=max_cores_per_head_batch
    )


def validate_recipe_args(
    precision: ttnn.SDPAPrecision | None,
    kv_dtype: ttnn.DataType | None,
    *,
    head_dim: int,
    model: str,
    is_blackhole: bool = True,
) -> ttnn.DataType:
    """Validate the resolved recipe and return the KV storage dtype (BF16 unless LOW_PRECISION asks otherwise)."""
    kv_dtype = kv_dtype or ttnn.bfloat16
    if precision is None:
        if kv_dtype != ttnn.bfloat16:
            raise ValueError(f"{model}: sdpa_kv_dtype requires sdpa_precision=LOW_PRECISION")
        return kv_dtype
    if not is_blackhole or head_dim not in RECIPE_HEAD_DIMS:
        raise ValueError(
            f"{model}: named SDPA recipes require Blackhole attention with head_dim in {RECIPE_HEAD_DIMS} "
            f"(head_dim={head_dim})"
        )
    if kv_dtype not in RECIPE_KV_DTYPES:
        raise ValueError(f"{model}: unsupported SDPA KV dtype {kv_dtype}")
    if kv_dtype != ttnn.bfloat16 and precision != ttnn.SDPAPrecision.LOW_PRECISION:
        raise ValueError(f"{model}: low-precision KV requires the LOW_PRECISION recipe")
    return kv_dtype


def exp_ring_supports(head_dim: int) -> bool:
    """Exp ring recipes are D128 only; other head dims must use ring joint SDPA."""
    return head_dim in EXP_RING_HEAD_DIMS


def recipe_program_config(
    program_config: ttnn.SDPAProgramConfig, *, ring: bool = False, exp_ring: bool = False
) -> ttnn.SDPAProgramConfig:
    """The same grid with op-selected chunks; exp_approx_mode is left unset (recipes own it).

    ``ring``/``exp_ring`` name the call site only: every recipe op picks its own blocking.
    """
    del ring, exp_ring
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=program_config.compute_with_storage_grid_size,
        max_cores_per_head_batch=program_config.max_cores_per_head_batch,
    )


def recipe_sdpa_kwargs(precision: ttnn.SDPAPrecision) -> dict:
    """Keyword arguments replacing ``compute_kernel_config`` on every recipe SDPA call."""
    return {"precision": precision, "inputs_prepared": precision == ttnn.SDPAPrecision.LOW_PRECISION}


def sdpa_kwargs(precision: ttnn.SDPAPrecision | None, compute_kernel_config) -> dict:
    """Recipe kwargs, or the legacy compute config (read by the caller at call time)."""
    if precision is not None:
        return recipe_sdpa_kwargs(precision)
    return {"compute_kernel_config": compute_kernel_config}


def prepare_recipe_inputs(precision: ttnn.SDPAPrecision | None, kv_dtype: ttnn.DataType, q, k, v):
    """LOW_PRECISION input preparation (after norm/RoPE, before any ring communication).

    Returns the inputs unchanged for every other recipe and for the legacy path. ``None`` entries
    (e.g. absent joint tensors) pass through.
    """
    if precision != ttnn.SDPAPrecision.LOW_PRECISION:
        return q, k, v
    prepare = ttnn.transformer.prepare_sdpa_input
    return (
        None if q is None else prepare(q, is_query=True),
        None if k is None else prepare(k, is_query=False, dtype=kv_dtype),
        None if v is None else prepare(v, is_query=False, dtype=kv_dtype),
    )
