# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Canonical host-weight schema for AttnRes: the checkpoint's key names and the fold into queries.

AttnRes holds no matrices — every one of its weights is a `[d]` vector, two per read site,
and they only ever meet as a product against the candidate (`reference/kimi_k3/attn_res/attn_res.py`).
So the load is a fold, not a conversion, and there is nothing here worth a `.tensorbin`
cache: all 374 vectors of a 93-layer stack come to 5 MB against the ~100 GB the matmul
weights put through `tt/WEIGHTS_AND_CACHE.md`'s three-method pattern.

Unlike the rest of the model's weights these are not layer-local. One `TtAttnRes` serves the
whole stack, so the unit a caller wants is the full set in walk order rather than a slice per
`TtPrefillBlock` — hence `fold_queries` returning the three lists `attn_res_stack` takes.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.attn_res import HIDDEN_SIZE, NUM_LAYERS, fold_query

# Kimi K3 nests the decoder under a multimodal wrapper, so its published checkpoint spells
# these `language_model.model.layers.…`. A state dict taken from an instantiated model is
# already rooted at the decoder and wants `prefix=""`.
CHECKPOINT_PREFIX = "language_model.model."

# The two factors of one query, in the order `fold_query` multiplies them.
QUERY_FACTORS = ("norm", "proj")

# One read site before self-attention and one before the MLP, named for what follows them.
SITE_PARTS = ("self_attention", "mlp")


def layer_query_names(layer_idx: int, prefix: str = CHECKPOINT_PREFIX) -> tuple[str, ...]:
    """Both factors of both of layer `layer_idx`'s queries."""
    return tuple(
        f"{prefix}layers.{layer_idx}.{part}_res_{factor}.weight" for part in SITE_PARTS for factor in QUERY_FACTORS
    )


def output_query_names(prefix: str = CHECKPOINT_PREFIX) -> tuple[str, ...]:
    """Both factors of the single model-level read, after the last layer."""
    return tuple(f"{prefix}output_attn_res_{factor}.weight" for factor in QUERY_FACTORS)


def query_weight_names(num_layers: int = NUM_LAYERS, prefix: str = CHECKPOINT_PREFIX) -> tuple[str, ...]:
    """Every AttnRes weight in the checkpoint, and nothing else."""
    per_layer = tuple(name for layer_idx in range(num_layers) for name in layer_query_names(layer_idx, prefix))
    return per_layer + output_query_names(prefix)


def validate_query_weights(
    weights: Mapping[str, torch.Tensor],
    num_layers: int = NUM_LAYERS,
    hidden_size: int = HIDDEN_SIZE,
    prefix: str = CHECKPOINT_PREFIX,
) -> None:
    """Check that every query weight is present and has the shape the model declares.

    `res_norm.weight` is `[d]` and `res_proj.weight` is `[1, d]` — the latter is a projection
    to a scalar score, so the checkpoint keeps the output dimension. Both shapes are read off
    the HuggingFace module definitions rather than an observed checkpoint.

    `fold_query` flattens both factors, so a transposed store would fold to the same numbers
    and pass a bare element count. Checking the shape is what makes this a load-time boundary
    instead of a silent reinterpretation.
    """
    for name in query_weight_names(num_layers, prefix):
        try:
            weight = weights[name]
        except KeyError as error:
            raise ValueError(f"missing AttnRes weight: {name}") from error
        expected = (1, hidden_size) if name.endswith(f"_{QUERY_FACTORS[1]}.weight") else (hidden_size,)
        if tuple(weight.shape) != expected:
            raise ValueError(f"{name} has shape {tuple(weight.shape)}, expected {expected}")


def fold_queries(
    weights: Mapping[str, torch.Tensor],
    num_layers: int = NUM_LAYERS,
    prefix: str = CHECKPOINT_PREFIX,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Fold the checkpoint's weight pairs into the queries the walk issues.

    Returns `(q_pre, q_post, q_out)` in `attn_res_stack`'s argument order. `q_pre[0]` is
    folded and returned for symmetry but never issued — layer 0 has nothing sealed to read
    against, and in the checkpoint that entry is a dead constant.
    """
    validate_query_weights(weights, num_layers, prefix=prefix)
    fold = lambda name: fold_query(weights[f"{name}_norm.weight"], weights[f"{name}_proj.weight"])
    site = lambda layer_idx, part: fold(f"{prefix}layers.{layer_idx}.{part}_res")
    return (
        [site(layer_idx, "self_attention") for layer_idx in range(num_layers)],
        [site(layer_idx, "mlp") for layer_idx in range(num_layers)],
        fold(f"{prefix}output_attn_res"),
    )
