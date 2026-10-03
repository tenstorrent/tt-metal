# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Opt-in precision knobs for the MiniMax-H3 transformer blocks, read from the environment: MINIMAX_H3_FAST=1
(FAST_RECIPE) and MINIMAX_H3_BF8_WEIGHTS=qkv,ff1[,out,ff2] (typecast those linears' weights to bfloat8_b after
loading). Documented in models/tt_dit/models/MiniMaxH3.md."""

from __future__ import annotations

import os

from loguru import logger

import ttnn

FAST_RECIPE = {
    "MINIMAX_H3_BF8_WEIGHTS": "qkv,ff1",
    "MINIMAX_H3_SDPA_PV_FIDELITY": "LoFi",
}


def apply_fast_recipe_env() -> None:
    """MINIMAX_H3_FAST=1 fills in the recipe knobs (explicit settings win); runs at import, before any module reads them."""
    if os.environ.get("MINIMAX_H3_FAST") == "1":
        for key, value in FAST_RECIPE.items():
            os.environ.setdefault(key, value)


apply_fast_recipe_env()

LINEARS = ("qkv", "out", "ff1", "ff2")


def math_fidelity_from_env(var: str) -> ttnn.MathFidelity | None:
    """The MathFidelity named by an environment variable; None when it is unset."""
    name = os.environ.get(var)
    if not name:
        return None
    valid = [key for key in ttnn.MathFidelity.__members__ if key != "Invalid"]
    if name not in valid:
        raise ValueError(f"{var}={name!r}: expected one of {valid}")
    return ttnn.MathFidelity.__members__[name]


def _typecast_parameter(param, dtype) -> None:
    # The declared dtype stays as cached so a reload after eviction passes the parameter's dtype check; the live
    # tensor is what gets cast, and the pipeline applies this again after every load.
    if param._data is not None and param._data.dtype != dtype:
        param._data = ttnn.typecast(param._data, dtype)


def apply_env_quant_config(model) -> None:
    """`model` is the transformer or a single block (the block perf test)."""
    bf8 = [name for name in os.environ.get("MINIMAX_H3_BF8_WEIGHTS", "").split(",") if name]
    if not bf8:
        return
    unknown = sorted(set(bf8) - set(LINEARS))
    if unknown:
        raise ValueError(f"MINIMAX_H3_BF8_WEIGHTS: unknown {unknown}, expected a subset of {list(LINEARS)}")
    logger.info(f"minimax-h3 block weights typecast to bfloat8_b: {bf8}")
    for block in getattr(model, "transformer_blocks", [model]):
        linears = {"qkv": block.attn.to_qkv, "out": block.attn.to_out, "ff1": block.ff.ff1, "ff2": block.ff.ff2}
        for name in bf8:
            _typecast_parameter(linears[name].weight, ttnn.bfloat8_b)
