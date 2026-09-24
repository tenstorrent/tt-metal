# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared opt-in SDPA precision recipe wiring for tt_dit denoisers.

Mirrors the Wan integration (models/transformers/wan2_2/attention_wan.py): a model constructor takes
``sdpa_precision: ttnn.SDPAPrecision | None`` and ``sdpa_kv_dtype: ttnn.DataType | None``. ``None``
keeps the model's existing attention configuration exactly; nothing in this module is consulted then.

Recipe blocking is op-selected (docs/sdpa_precision.md, "Blocking"): a recipe config keeps the
caller's grid and leaves ``q_chunk_size``/``k_chunk_size`` at 0, and SDPA chooses the chunks (and, for
exp ring, the grid width) from the shape, recipe, op, grid and L1. The legacy path keeps its tuned
configs unchanged. Exp ring joint SDPA recipes are D128 only.
"""

from __future__ import annotations

import ttnn

RECIPE_HEAD_DIMS = (64, 128, 256)
EXP_RING_HEAD_DIMS = (128,)
RECIPE_KV_DTYPES = (ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b)


def validate_recipe_args(
    precision: ttnn.SDPAPrecision | None,
    kv_dtype: ttnn.DataType | None,
    *,
    head_dim: int,
    model: str,
    is_blackhole: bool = True,
) -> ttnn.DataType:
    """Validate the opt-in and return the KV storage dtype (BF16 unless LOW_PRECISION asks otherwise)."""
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


def reject_mask(precision: ttnn.SDPAPrecision | None, mask, *, model: str) -> None:
    """Named recipes support unmasked attention only."""
    if precision is not None and mask is not None:
        raise ValueError(f"{model}: named SDPA recipes support unmasked attention only")
