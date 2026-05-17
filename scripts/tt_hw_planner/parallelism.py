# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Parallelism — how memory is divided across chips.

Sharding rules:

  Tensor Parallelism (TP):  weights and activations are sharded along the
    hidden-dim axis ÷ tp.  Most modern transformers do this.  CCL all-gather
    overhead is accounted for in hardware.py overhead constants.

  Pipeline Parallelism (PP): the model is split along the LAYER axis ÷ pp.
    Each chip holds 1/pp of the layers, all of their weights, and all of
    the KV cache for those layers.  Per-stage activations (the pipeline
    "bubble") add a 1× residual-stream copy per chip.

  Expert Parallelism (EP): MoE-only.  The N experts are sharded ÷ ep.
    Attention layers' weights still replicate / shard via TP.  Phase 2 keeps
    this conservative: until full expert sharding is wired in, we evaluate
    EP=1.  TP × PP enumeration covers the common bring-up choices.

  Data Parallelism (DP): not modelled (replicates every parameter; doesn't
    save memory).

The total chip count is tp × pp × ep × dp.  Search enumerates only
combinations whose product equals the mesh's chip count.

Empirical "replicated weights fraction" (_REPLICATED_FRAC) covers
embedding, lm_head, norms — these aren't TP-sharded in most ports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

from .architecture import MemoryModel
from .hardware import Box


# Fraction of model weights that are NOT TP-sharded (embedding, lm_head,
# norms).  Empirical figure based on Llama-3-style architectures.
_REPLICATED_FRAC = 0.04


@dataclass(frozen=True)
class ParallelConfig:
    """A specific parallelism configuration for a mesh."""

    tp: int
    pp: int = 1
    ep: int = 1
    dp: int = 1

    @property
    def chips(self) -> int:
        return self.tp * self.pp * self.ep * self.dp

    @property
    def label(self) -> str:
        if self.pp == self.ep == self.dp == 1:
            return f"TP={self.tp}"
        parts = [f"TP={self.tp}"]
        if self.pp != 1:
            parts.append(f"PP={self.pp}")
        if self.ep != 1:
            parts.append(f"EP={self.ep}")
        if self.dp != 1:
            parts.append(f"DP={self.dp}")
        return ",".join(parts)


def _divisors(n: int):
    out = []
    for i in range(1, n + 1):
        if n % i == 0:
            out.append(i)
    return out


def canonical_meshes(box: Box) -> List[Tuple[int, int]]:
    """One canonical mesh per chip-count for this box.  Largest TP comes last."""
    seen = {}
    for shape in box.mesh_shapes:
        chips = shape[0] * shape[1]
        if chips not in seen:
            seen[chips] = shape
    return list(seen.values())


def enumerate_parallelism(chips: int, explore_pp: bool = False) -> List[ParallelConfig]:
    """
    Yield ParallelConfigs whose total chip count equals `chips`.

    Pure-TP mode (default): just one config, TP=chips, PP=1.
    explore_pp=True:        every (TP, PP) such that TP×PP=chips.
    """
    if not explore_pp or chips == 1:
        return [ParallelConfig(tp=chips)]
    out = []
    for tp in _divisors(chips):
        pp = chips // tp
        out.append(ParallelConfig(tp=tp, pp=pp))
    return out


def enumerate_meshes(box: Box, explore_pp: bool = False) -> Iterator[Tuple[Tuple[int, int], ParallelConfig]]:
    """
    Yield (mesh_shape, parallel_config) for each canonical mesh on `box`.

    Canonical = one mesh shape per chip-count.  For each, we enumerate the
    parallelism configurations (pure-TP or TP×PP, depending on explore_pp).
    """
    for shape in canonical_meshes(box):
        chips = shape[0] * shape[1]
        for pcfg in enumerate_parallelism(chips, explore_pp=explore_pp):
            yield shape, pcfg


@dataclass
class ShardedMemory:
    """Memory required on a single chip after applying parallelism."""

    weights_bytes: int
    kv_cache_bytes: int
    activation_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.weights_bytes + self.kv_cache_bytes + self.activation_bytes


def shard(
    model: MemoryModel, dtype: str, batch: int, seq: int, kv_dtype_bytes: float, pcfg: ParallelConfig
) -> ShardedMemory:
    """
    Apply (TP, PP) sharding to a MemoryModel and return per-chip byte counts.

    Model-level sizes are first split along the layer axis by PP, then the
    remaining per-stage sizes are split along the hidden axis by TP.
    """
    full_weights = model.weights_bytes(dtype)
    full_kv = model.kv_cache_bytes(batch, seq, kv_dtype_bytes)
    full_act = model.activation_bytes(batch, seq, dtype="bf16")

    tp = max(pcfg.tp, 1)
    pp = max(pcfg.pp, 1)
    arch = model.arch

    # PP splits along layers: weights and KV both scale ÷ pp.
    # Activations gain a small per-stage replication penalty (the
    # pipeline bubble) — we model that as 1× residual stream per chip
    # not divided by PP.
    stage_weights = full_weights // pp
    stage_kv = full_kv // pp
    stage_act = full_act  # bubble: each PP stage keeps full per-token activations

    # TP splits each PP stage's tensors along hidden / kv_heads axes.
    # Embedding / lm_head / norms are still replicated per TP group.
    replicated_w = int(stage_weights * _REPLICATED_FRAC)
    sharded_w = (stage_weights - replicated_w) // tp
    per_chip_w = replicated_w + sharded_w

    # KV cache is sharded along kv_heads.  When tp exceeds the number of
    # kv_heads, sharding stops helping (each chip must duplicate at least
    # one head).
    effective_kv_shards = min(tp, max(arch.num_key_value_heads, 1))
    per_chip_kv = stage_kv // effective_kv_shards

    per_chip_act = stage_act // tp

    return ShardedMemory(
        weights_bytes=per_chip_w,
        kv_cache_bytes=per_chip_kv,
        activation_bytes=per_chip_act,
    )
