# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The WHOLE-MODEL reference forward, and its on-disk cache.

Torch only.

## Why this is cached when the per-module goldens are not

A full-depth CPU forward of an 8B model is expensive — minutes, not seconds — and several tests want
the same one. The per-module goldens of `golden.py` are cheap enough to recompute on a miss; this one
is not, so it follows the opposite rule: **assert rather than recompute** where a test that silently
regenerates would burn the run instead of failing.

The cache machinery is IMPORTED from `deepseek_v3_d_p/utils/transformer_helpers.py` rather than
reimplemented — `ReferenceCacheKey`, `save_reference_cache`, `load_reference_cache` — because the
property that matters is one this bring-up must not weaken: the key is a FROZEN dataclass covering
every field that changes the output, so a changed field yields a different filename and a stale
result is never silently reused.

`ReferenceCacheKey` carries `n_routed_experts` because it was written for the MoE family. This model
is dense, so it is always **0** — a real value that distinguishes this model's cache from a MoE
sibling's, not a placeholder.

`variant` in those helpers is duck-typed: it needs only `.name` and `.ref_cache_env`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from loguru import logger

from models.demos.deepseek_v3_d_p.utils.transformer_helpers import (
    ReferenceCacheKey,
    load_reference_cache,
    save_reference_cache,
)

from .config import LlamaConfigConstants
from .model import REF_DTYPE, RefModel


@dataclass(frozen=True)
class LlamaVariant:
    """The `variant` handle the imported cache helpers expect. Two fields, both used for the path."""

    name: str = "llama_3_1_8b"
    ref_cache_env: str = "TT_LLAMA31_8B_PREFILL_HOST_REF_CACHE"


VARIANT = LlamaVariant()
N_ROUTED_EXPERTS = 0  # dense model — no experts. A real value, not a placeholder.


def cache_key(
    *,
    weight_type: str,
    input_source: str,
    isl_total: int,
    num_layers: int,
    padding_side: str = "right",
) -> ReferenceCacheKey:
    """Build the frozen cache key. `weight_type` is "pretrained" or "random"."""
    assert weight_type in ("pretrained", "random"), weight_type
    return ReferenceCacheKey(
        weight_type=weight_type,
        input_source=input_source,
        isl_total=isl_total,
        num_layers=num_layers,
        n_routed_experts=N_ROUTED_EXPERTS,
        padding_side=padding_side,
    )


def run_reference_forward(
    model: RefModel,
    input_ids: torch.Tensor,
) -> tuple[list[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]]]:
    """One full forward, returning per-layer hidden-state snapshots and per-layer (K, V).

    The KV list is the graded artifact: P1/P2 feed the same token ids to the device and PCC its KV
    cache against these tensors, layer by layer.
    """
    snapshots: list[torch.Tensor] = []
    kv: list[tuple[torch.Tensor, torch.Tensor]] = []

    b, s = input_ids.shape
    with torch.no_grad():
        hidden = model.embed_tokens(input_ids)
        position_ids = torch.arange(s)[None, :].expand(b, -1)
        cos, sin = model.rotary_emb(position_ids, dtype=hidden.dtype)
        from .model import causal_mask

        mask = causal_mask(s, hidden.dtype)
        for i, layer in enumerate(model.layers):
            hidden, k_rope, v = layer(hidden, (cos, sin), mask, return_kv=True)
            snapshots.append(hidden.clone())
            kv.append((k_rope.clone(), v.clone()))
            logger.debug(f"reference layer {i}/{len(model.layers)}")
    return snapshots, kv


def get_reference(
    config: LlamaConfigConstants,
    input_ids: torch.Tensor,
    *,
    weight_type: str,
    input_source: str,
    num_layers: int | None = None,
    state_dict: dict | None = None,
    allow_recompute: bool = True,
):
    """Load the cached whole-model reference, or compute and save it.

    Args:
        allow_recompute: False makes a cache MISS an error instead of a multi-minute recompute.
            That is what CI should pass: a test that quietly regenerates the reference turns a
            missing artifact into a timeout rather than a clear failure.
    """
    num_layers = config.num_hidden_layers if num_layers is None else num_layers
    key = cache_key(
        weight_type=weight_type,
        input_source=input_source,
        isl_total=int(input_ids.shape[-1]),
        num_layers=num_layers,
    )
    try:
        return load_reference_cache(VARIANT, key)
    except FileNotFoundError:
        if not allow_recompute:
            raise FileNotFoundError(
                f"whole-model reference cache miss for {key}, and recompute is disabled. Generate it "
                f"first (a full-depth CPU forward takes minutes) rather than paying for it here."
            ) from None

    logger.info(f"computing whole-model reference for {key} (this is the expensive one)")
    model = RefModel(config, num_layers=num_layers).to(REF_DTYPE).eval()
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)
    snapshots, kv = run_reference_forward(model, input_ids)
    # The imported saver stores (snapshots, kvpe) pairs; for a GQA model the second slot holds the
    # (K, V) tuples rather than MLA's single latent row.
    save_reference_cache(VARIANT, key, snapshots, kv)
    return snapshots, kv
