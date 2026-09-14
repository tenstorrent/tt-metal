# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compute-once cache for the expensive whole-model CPU reference forward.

The DeepSeek golden-cache helpers are **imported, not copied** (recipe §3): ``ReferenceCacheKey``
(frozen, so a changed field yields a different filename and a stale result is never silently reused)
plus ``save_reference_cache`` / ``load_reference_cache`` / ``check_reference_cache_exists``.

Two adaptations, both at the edges rather than in the helpers:

* those functions take a DeepSeek ``variant`` object for the cache directory and env var, so
  :data:`VARIANT` is a tiny stand-in carrying just the two attributes they read;
* the payload slots are named for MLA (``ref_snapshots`` / ``ref_kvpe_list``). A GQA model has a K
  and a V rather than one latent line, so the KV slot holds ``torch.stack([k, v])`` per layer and
  :func:`unpack_kv` is the only place that is unpacked. ``n_routed_experts`` in the key is 0 for a
  dense model — the field is kept because dropping it would mean forking the frozen key class, which
  is exactly what the "import, never copy" rule exists to prevent.

The two rules the cache embodies, which matter more than the mechanism: key on **every** field that
changes the output, and **assert rather than recompute** where a CPU run is expensive — a cache miss
in CI should fail loudly in a second, not burn an hour silently.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import List, Tuple

import torch

from models.demos.deepseek_v3_d_p.utils.transformer_helpers import (
    ReferenceCacheKey,
    check_reference_cache_exists,
    load_reference_cache,
    save_reference_cache,
)

VARIANT = SimpleNamespace(name="llama_3_1_8b", ref_cache_env="LLAMA_PREFILL_HOST_REF_CACHE")


def cache_key(*, weight_type: str, input_source: str, isl_total: int, num_layers: int) -> ReferenceCacheKey:
    """Every field that changes the reference's output. ``n_routed_experts`` is 0: Llama is dense."""
    return ReferenceCacheKey(
        weight_type=weight_type,
        input_source=input_source,
        isl_total=isl_total,
        num_layers=num_layers,
        n_routed_experts=0,
        padding_side="right",
    )


def exists(key: ReferenceCacheKey) -> bool:
    return check_reference_cache_exists(VARIANT, key)


def save(key: ReferenceCacheKey, hidden_states: List[torch.Tensor], kv: List[Tuple[torch.Tensor, torch.Tensor]]):
    """``kv`` is one ``(k, v)`` per layer; stacked into the single-tensor slot the helper expects."""
    save_reference_cache(VARIANT, key, hidden_states, [torch.stack([k, v]) for k, v in kv])


def load(key: ReferenceCacheKey):
    snapshots, packed = load_reference_cache(VARIANT, key)
    return snapshots, unpack_kv(packed)


def unpack_kv(packed: List[torch.Tensor]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    return [(t[0], t[1]) for t in packed]


def cache_dir() -> str:
    return os.environ.get(VARIANT.ref_cache_env, f"/tmp/{VARIANT.name}_transformer_ref_cache")
