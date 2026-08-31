# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Apply a parsed adapter to a built model.

The whole loader is three steps, and only the first carries any model-specific knowledge:

1. **Group.** A model that fuses several checkpoint projections into one Linear needs their
   adapters fused the same way. The group declaration says which adapter leaves share a
   destination and in what order -- for MiniMax-H3 that is q/k/v into ``to_qkv``, and nothing else.
2. **Route.** Hand the ``B`` factors and the full-rank deltas to :func:`.route.route`, which walks
   the model's own ``_prepare_torch_state`` pipeline and hands back each destination parameter with
   its tensor in final layout. Head interleaving, rotary channel permutation and SwiGLU tile
   packing all happen there, in the model's code, not here.
3. **Bind.** Low-rank factors go into the target Linear's adapter bank and get bound; full-rank
   deltas are added straight into the parameter.

Fusing a group multiplies the bound rank by the group size: ``A`` stacks along rank and ``B``
becomes block diagonal, so ``B_fused @ A_fused`` reproduces the per-source deltas exactly and
nothing else. Rank 64 q/k/v therefore binds as rank 192. That is the cost of the fused projection
and it is paid once, at bind, not per forward.

Nothing here writes to the base weight cache. The adapter is applied after ``cache.load_model``, so
one cached copy of the base weights serves every adapter and every strength.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from loguru import logger

from ..layers.lora import LoRAMixin
from .direct import apply_direct_delta
from .keys import parse_adapter
from .route import route

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from ..layers.module import Module
    from .direct import DirectDelta
    from .keys import AdapterEntry


@dataclass(frozen=True)
class FusionGroup:
    """Adapter leaves that share one destination parameter.

    ``members`` is ordered, and the order is load-bearing: it fixes which rank columns of the
    block-diagonal ``B`` belong to which source, and must match the order the model's own
    ``_prepare_torch_state`` concatenates them in.
    """

    owner: str
    """Dotted adapter-space path of the module holding the members, e.g. ``...blocks.0.attn``."""
    members: tuple[str, ...]
    """Member names relative to ``owner``, e.g. ``("to_q", "to_k", "to_v")``."""


@dataclass
class AdapterReport:
    """What was applied, and to what. Every parsed entry lands in exactly one bucket."""

    name: str
    strength: float
    bound: dict[str, int] = field(default_factory=dict)
    """Destination parameter path -> bank index of the bound adapter."""
    deltas: list[DirectDelta] = field(default_factory=list)
    host: list[AdapterEntry] = field(default_factory=list)
    """Entries the model does not hold on device; the caller's ``host_sink`` owns these."""
    rejected: dict[str, str] = field(default_factory=dict)
    """Adapter path -> why it cannot be applied. Non-empty means the adapter is not fully applied."""

    def summary(self) -> str:
        return (
            f"{self.name}: bound {len(self.bound)} low-rank, added {len(self.deltas)} dense, "
            f"deferred {len(self.host)} to host, rejected {len(self.rejected)}"
        )


def apply_adapter(
    model: Module,
    source: str | Path | Mapping[str, torch.Tensor],
    *,
    groups: Sequence[FusionGroup] = (),
    is_host: Callable[[str], bool] | None = None,
    strength: float = 1.0,
    name: str = "",
    expect: Mapping[str, int] | None = None,
) -> AdapterReport:
    """Parse an adapter and load it onto ``model``. See :func:`apply_entries` for the arguments.

    ``expect`` asserts per-kind tensor counts, normally taken from the adapter's own metadata. A
    coverage check is the only thing that catches a target map that quietly misses a subset of the
    adapter, because output parity against a reference built the same way cannot.
    """
    entries, stats = parse_adapter(source)
    logger.info(f"adapter {name or source}: {stats}")
    if expect is not None:
        _assert_counts(stats, expect)
    return apply_entries(model, entries, groups=groups, is_host=is_host, strength=strength, name=name or str(source))


def apply_entries(
    model: Module,
    entries: Sequence[AdapterEntry],
    *,
    groups: Sequence[FusionGroup] = (),
    is_host: Callable[[str], bool] | None = None,
    strength: float = 1.0,
    name: str = "",
) -> AdapterReport:
    """Load already-parsed adapter entries onto ``model``.

    Split out from :func:`apply_adapter` so a caller that needs the host half separately -- the
    MiniMax-H3 pipeline folds it into the AdaLN precompute -- parses the file once instead of
    twice; the published adapters are 1.5 to 5.3 GB.

    ``is_host`` marks adapter paths the model deliberately does not hold as device parameters --
    MiniMax-H3 precomputes its AdaLN projections on host, so their adapters have no module to bind
    to and are handed back for the caller to fold in elsewhere. Marking a path is an assertion that
    someone else applies it; failing to mark one surfaces as an unresolved key rather than as
    silence.

    """
    report = AdapterReport(name=name, strength=strength)
    device_entries = []
    for entry in entries:
        if entry.kind == "set_weight":
            report.rejected[entry.path] = "set_weight replaces a parameter absent from the base architecture"
        elif is_host is not None and is_host(entry.path):
            report.host.append(entry)
        else:
            device_entries.append(entry)

    if report.rejected:
        paths = ", ".join(sorted(report.rejected)[:3])
        msg = (
            f"adapter {report.name} carries {len(report.rejected)} unsupported payload(s) ({paths}...); "
            "this adapter needs an architecture the port does not have"
        )
        raise NotImplementedError(msg)

    fused, singles = _partition(device_entries, groups)
    state, provenance = _build_routing_state(fused, singles)
    routed, unresolved = route(model, state)
    if unresolved:
        msg = (
            f"adapter {report.name}: {len(unresolved)} target(s) resolved to no parameter: "
            f"{', '.join(sorted(unresolved)[:5])}. Either the target map is incomplete or this "
            "adapter was published against a different architecture."
        )
        raise KeyError(msg)

    for item in routed:
        sources = provenance[item.path]
        if sources[0].kind == "lora":
            report.bound[item.path] = _bind(item.module, item.value, sources, strength=strength, name=report.name)
        else:
            report.deltas.append(apply_direct_delta(item.path, item.param, item.value, strength=strength))

    logger.info(report.summary())
    return report


def _assert_counts(stats, expect: Mapping[str, int]) -> None:
    mismatched = {k: (stats.counts.get(k, 0), v) for k, v in expect.items() if stats.counts.get(k, 0) != v}
    if mismatched:
        detail = ", ".join(f"{k}: parsed {got}, expected {want}" for k, (got, want) in sorted(mismatched.items()))
        msg = f"adapter tensor counts disagree with its metadata ({detail})"
        raise ValueError(msg)


def _partition(
    entries: Sequence[AdapterEntry], groups: Sequence[FusionGroup]
) -> tuple[dict[str, list[AdapterEntry]], list[AdapterEntry]]:
    """Split entries into fused groups (keyed by owner) and standalone ones.

    A group whose members are only partly present is fatal: the model's fusing hook reads all of
    them, and a zero-filled stand-in would be a silent change to the delta rather than an omission.
    """
    member_owner = {f"{g.owner}.{m}": (g, m) for g in groups for m in g.members}
    fused: dict[str, list[AdapterEntry]] = {}
    singles: list[AdapterEntry] = []
    for entry in entries:
        found = member_owner.get(entry.path)
        if found is None:
            singles.append(entry)
        else:
            fused.setdefault(found[0].owner, []).append(entry)

    for owner, members in fused.items():
        group = next(g for g in groups if g.owner == owner)
        present = {entry.path.rsplit(".", 1)[-1] for entry in members}
        missing = [m for m in group.members if m not in present]
        if missing:
            msg = f"{owner}: fused group is missing {missing}; it must be adapted whole or not at all"
            raise ValueError(msg)
        kinds = {entry.kind for entry in members}
        if kinds != {"lora"}:
            msg = f"{owner}: fused groups support low-rank members only, got {sorted(kinds)}"
            raise ValueError(msg)
        members.sort(key=lambda e: group.members.index(e.path.rsplit(".", 1)[-1]))

    return fused, singles


def _build_routing_state(
    fused: Mapping[str, list[AdapterEntry]], singles: Sequence[AdapterEntry]
) -> tuple[dict[str, torch.Tensor], _Provenance]:
    """Assemble the state dict handed to the router, and remember what produced each key.

    Fused members are widened to the group's total rank before routing, each occupying its own
    slice of the rank axis. That block-diagonal form is what makes the fused product reproduce the
    per-source deltas and cross-couple nothing.

    Provenance is keyed by the *unprefixed* destination name because the router reports paths from
    the model root, and for a fused group the destination is a parameter no source is named after.
    """
    state: dict[str, torch.Tensor] = {}
    by_source: dict[str, list[AdapterEntry]] = {}

    for owner, members in fused.items():
        total_rank = sum(entry.rank for entry in members)
        offset = 0
        for entry in members:
            widened = torch.zeros(entry.B.shape[0], total_rank, dtype=entry.B.dtype)
            widened[:, offset : offset + entry.rank] = entry.B
            offset += entry.rank
            state[f"{entry.path}.weight"] = widened
        by_source[owner] = members

    for entry in singles:
        suffix = "bias" if entry.kind == "diff_b" else "weight"
        state[f"{entry.path}.{suffix}"] = entry.B if entry.kind == "lora" else entry.delta
        by_source[entry.path] = [entry]

    return state, _Provenance(by_source)


class _Provenance:
    """Map a routed destination path back to the adapter entries that produced it.

    The router reports where a tensor landed, not where it came from. For a 1:1 target the two
    share a prefix; for a fused group the destination sits under the owner. Longest-prefix match
    covers both without the router having to carry provenance itself.
    """

    def __init__(self, by_source: Mapping[str, list[AdapterEntry]]) -> None:
        self._by_source = dict(by_source)

    def __getitem__(self, dest: str) -> list[AdapterEntry]:
        candidates = [src for src in self._by_source if dest == src or dest.startswith(f"{src}.")]
        if not candidates:
            msg = f"routed tensor {dest} has no adapter entry behind it"
            raise KeyError(msg)
        return self._by_source[max(candidates, key=len)]


def _bind(module: Module, b_routed: torch.Tensor, sources: Sequence[AdapterEntry], *, strength: float, name: str):
    """Register the routed ``B`` with its stacked ``A`` on ``module`` and make it active.

    Routing leaves ``B`` in the parameter's own ``[rank, out]`` orientation, since that is what the
    preparation pipeline produces for a weight. The bank stores factors in PyTorch LoRA layout, so
    it goes back to ``[out, rank]`` here rather than teaching the mixin a second convention.
    """
    if not isinstance(module, LoRAMixin):
        msg = (
            f"{name}: {type(module).__name__} is not LoRA-aware; call promote_to_lora(model) before "
            "applying an adapter"
        )
        raise TypeError(msg)

    a_stacked = torch.cat([entry.A for entry in sources], dim=0)
    # One scale for the group: distinct per-source alphas cannot be expressed once the sources
    # share a bound adapter, and no published adapter has produced them.
    scales = {entry.scale for entry in sources}
    if len(scales) != 1:
        msg = f"{name}: fused group members disagree on scale {sorted(scales)}"
        raise ValueError(msg)

    b_bank = b_routed.transpose(0, 1).contiguous()
    idx = module.register_lora(a_stacked, b_bank, scale=strength * scales.pop(), name=name, prepared=True)
    module.bind_active(idx)
    return idx
