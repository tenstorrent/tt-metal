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

  Sequence Parallelism (SP): one REQUEST'S TOKENS are cut into equal,
    tile-aligned slices, one slice per chip group.  The groups replicate
    weights like DP replicas but each works on its own slice of the same
    request, exchanging only attention K/V -- so KV cache and activations
    divide by sp while weights do not.  It takes the chips DP would leave
    idle when a run serves fewer concurrent requests than it has spare
    chips; the rule is agent/tp.py:split_spare_chips, and a run that states
    no workload keeps sp=1.

The total chip count is tp × pp × ep × dp × sp.  Search enumerates only
combinations whose product equals the mesh's chip count.

Empirical "replicated weights fraction" (_REPLICATED_FRAC) covers
embedding, lm_head, norms — these aren't TP-sharded in most ports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

from .architecture import MemoryModel
from .hardware import Box


PARALLELISM_MANIFEST = "parallelism_manifest.json"


_REPLICATED_FRAC = 0.04


@dataclass(frozen=True)
class ParallelConfig:
    """A specific parallelism configuration for a mesh."""

    tp: int
    pp: int = 1
    ep: int = 1
    dp: int = 1
    sp: int = 1

    @property
    def chips(self) -> int:
        return self.tp * self.pp * self.ep * self.dp * self.sp

    @property
    def label(self) -> str:
        if self.pp == self.ep == self.dp == self.sp == 1:
            return f"TP={self.tp}"
        parts = [f"TP={self.tp}"]
        if self.pp != 1:
            parts.append(f"PP={self.pp}")
        if self.ep != 1:
            parts.append(f"EP={self.ep}")
        if self.dp != 1:
            parts.append(f"DP={self.dp}")
        if self.sp != 1:
            parts.append(f"SP={self.sp}")
        return ",".join(parts)


def split_label(tp: int, dp: int, sp: int = 1) -> str:
    """The one spelling of a split the operator-facing lines share ("TP=2 x DP=2", "TP=1 x DP=1 x SP=4").
    SP is named only when it is in play, so every line that never saw SP reads exactly as before."""
    out = f"TP={int(tp)} x DP={int(dp)}"
    if int(sp or 1) > 1:
        out += f" x SP={int(sp)}"
    return out


def _divisors(n: int):
    out = []
    for i in range(1, n + 1):
        if n % i == 0:
            out.append(i)
    return out


def canonical_meshes(box: Box) -> List[Tuple[int, int]]:
    """Every canonical mesh shape declared for this box.

    Previously this deduplicated by chip count (kept only the largest-TP
    shape per chip-count), which silently hid same-chip-count alternatives
    like [2,2] / [4,1] from the verdict pipeline. Those alternatives
    matter when the largest-TP shape fails kernel divisibility for a
    given model's head counts; the verdict layer is then free to bump the
    recommendation to a shape that the model actually fits.

    `pick_best` (in verdict.py) tiebreaks toward fewest chips when
    everything else is equal, so emitting the full list does not change
    the default recommendation for the common case.
    """
    return list(box.mesh_shapes)


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


def _fill_spare(tp: int, spare: int, requests, seq_len) -> ParallelConfig:
    """DP x SP for the `spare` chips a TP group leaves over.

    The rule lives in ONE place, the engine's agent/tp.py:split_spare_chips, because the optimize route
    applies the same rule to the same two facts. Without a stated workload every spare chip is a replica,
    exactly as before; the same holds when the engine package is not importable (a tree carrying only
    this planner), so SP can only ever be chosen where the engine that runs it is present."""
    dp, sp = spare, 1
    if requests is not None and seq_len is not None:
        try:
            from models.experimental.perf_automation.agent.tp import split_spare_chips
        except ImportError:
            split_spare_chips = None
        if split_spare_chips is not None:
            dp, sp = split_spare_chips(spare, requests, seq_len)
    return ParallelConfig(tp=tp, dp=dp, sp=sp)


def select_parallelism(chips: int, kernel_report, *, requests=None, seq_len=None) -> ParallelConfig:
    """Turn per-TP kernel viability into a chosen TP x DP x SP split for `chips`.

    The tool computes viability per TP degree (KernelReport.has_blockers(tp) over tp_grid) but never
    acted on it — enumerate_parallelism only ever fills the mesh with TP and DP stays 1. This selector
    closes that gap, model-agnostically and engine-neutrally (both fsm and cc consume it upstream of the
    bring-up loop):

      tp = largest degree in the report's grid that divides `chips` AND has no kernel blockers
      the spare chips (chips // tp) become replicas (dp), or -- when the run states fewer concurrent
      `requests` than spare chips and a `seq_len` that cuts into tile-aligned slices -- groups that
      split one request's tokens (sp), via _fill_spare. Unstated workload: dp = chips // tp as before.

    Falls back to TP=1 if no larger degree is viable (TP=1 always divides and is the safe floor).
    Returns a ParallelConfig with tp, dp and sp set; tp * dp * sp == chips."""
    if chips <= 1:
        return ParallelConfig(tp=1, dp=1)
    grid = list(getattr(kernel_report, "tp_grid", None) or [1])
    candidates = sorted({tp for tp in grid if tp >= 1 and chips % tp == 0}, reverse=True)
    for tp in candidates:
        try:
            blocked = kernel_report.has_blockers(tp=tp)
        except Exception:
            blocked = True
        if not blocked:
            return _fill_spare(tp, chips // tp, requests, seq_len)
    return _fill_spare(1, chips, requests, seq_len)


def plan_parallelism(model_id: str, chips: int, *, requests=None, seq_len=None):
    """Shared topology planner for BOTH emit-e2e and optimize: probe the model, evaluate per-TP kernel
    viability, and return the select_parallelism ParallelConfig for `chips`. Returns None when chips<=1
    or the model cannot be probed (caller then runs single-chip / a 1D default). Engine-neutral: the
    only place either path decides a TP x DP x SP split, so both stay consistent. `requests` and
    `seq_len` are the workload the run states (see select_parallelism); unstated keeps sp=1."""
    if not model_id or not chips or chips <= 1:
        return None
    try:
        from .cli import evaluate_kernels, probe_model

        probe = probe_model(model_id)
        if not getattr(probe, "raw_config", None):
            return None
        kr = evaluate_kernels(probe.raw_config, tp_grid=None)
        return select_parallelism(chips, kr, requests=requests, seq_len=seq_len)
    except Exception:  # noqa: BLE001
        return None


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
    Apply (TP, PP, SP) sharding to a MemoryModel and return per-chip byte counts.

    Model-level sizes are first split along the layer axis by PP, then the
    remaining per-stage sizes are split along the hidden axis by TP.  SP cuts
    the sequence: each group holds its slice of the KV cache and activations
    while the weights stay whole on every group (sp=1 changes nothing).
    """
    full_weights = model.weights_bytes(dtype)
    full_kv = model.kv_cache_bytes(batch, seq, kv_dtype_bytes)
    full_act = model.activation_bytes(batch, seq, dtype="bf16")

    tp = max(pcfg.tp, 1)
    pp = max(pcfg.pp, 1)
    sp = max(getattr(pcfg, "sp", 1), 1)
    arch = model.arch

    stage_weights = full_weights // pp
    stage_kv = full_kv // pp
    stage_act = full_act

    replicated_w = int(stage_weights * _REPLICATED_FRAC)
    sharded_w = (stage_weights - replicated_w) // tp
    per_chip_w = replicated_w + sharded_w

    effective_kv_shards = min(tp, max(arch.num_key_value_heads, 1))
    per_chip_kv = stage_kv // effective_kv_shards // sp

    per_chip_act = stage_act // tp // sp

    return ShardedMemory(
        weights_bytes=per_chip_w,
        kv_cache_bytes=per_chip_kv,
        activation_bytes=per_chip_act,
    )


def write_parallelism_manifest(demo_dir, *, chips: int, tp: int, dp: int, sp: int = 1) -> Optional[Path]:
    """Persist the TOPOLOGY bring-up graduated at, so emit-e2e can hard-assert consistency instead of
    silently recomputing from its own --mesh. Records the decidable degrees only (chips/tp/dp, sp when
    it is in play, + the MeshShape(dp*sp, tp) it implies: the rows carry the replicas AND the groups that
    split one request's tokens); the per-component SCHEME stays in the graduated stub code for the LLM
    to read. A manifest written without SP is byte-identical to before. Best-effort: returns the path
    on success, None on any write failure."""
    path = Path(demo_dir) / PARALLELISM_MANIFEST
    data = {
        "chips": int(chips),
        "tp": int(tp),
        "dp": int(dp),
        "mesh": [int(dp) * int(sp or 1), int(tp)],
    }
    if int(sp or 1) != 1:
        data["sp"] = int(sp)
    try:
        path.write_text(json.dumps(data, indent=2) + "\n")
        return path
    except OSError:
        return None


def read_parallelism_manifest(demo_dir) -> Optional[dict]:
    """Load the graduated-topology manifest for `demo_dir`, or None if absent/unreadable/malformed."""
    path = Path(demo_dir) / PARALLELISM_MANIFEST
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or "chips" not in data:
        return None
    return data
