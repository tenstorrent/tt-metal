# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Expert layout on an expert-parallel mesh, and what a lossless placement is allowed to change.

Layout (shared by every expert-parallel MoE this tool supports): routed experts are spread evenly over a
``rows x groups`` mesh in column-major order,

    global expert id = (group * rows + row) * experts_per_chip + slot,

where a *group* is one mesh column (one dispatch group: tokens are dispatched along the rows of their own
column) and a *row* is one chip of that column. A placement is a permutation that may move an expert to
any row of its own group, never across groups, so dispatch groups, the expert count per chip and every
weight byte are unchanged. Tokens still select the same logical experts; only the chip that runs them moves.

Two equivalent encodings are used:
  * owners[e]  = row that logical expert ``e`` lives on (what the optimizer works with);
  * physical_to_logical[p] = logical expert stored in physical slot ``p`` (what caches and runtimes use).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass(frozen=True)
class MoeSpec:
    """What the host side needs to know about one model on one mesh."""

    name: str
    num_experts: int
    top_k: int
    rows: int = 8  # chips per dispatch group (mesh axis along which tokens are dispatched)
    groups: int = 4  # dispatch groups (mesh columns)
    transport: str = "ring"  # dispatch/combine route model, see transport.py
    num_links: int = 2
    chunk_tokens: int = 5120  # tokens per prefill chunk, split evenly over the rows
    n_chunks: int = 11  # chunks per timed sequence
    moe_layers: tuple = ()  # global layer ids that carry routed experts in the measured depth (informational)
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        chips = self.rows * self.groups
        if self.num_experts <= 0 or self.num_experts % chips:
            raise ValueError(
                f"{self.name}: {self.num_experts} experts do not spread evenly over {self.rows}x{self.groups}"
            )
        if not 0 < self.top_k <= self.num_experts:
            raise ValueError(f"{self.name}: invalid top_k {self.top_k}")
        if self.chunk_tokens % self.rows:
            raise ValueError(f"{self.name}: chunk of {self.chunk_tokens} tokens does not split over {self.rows} rows")

    @property
    def chips(self) -> int:
        return self.rows * self.groups

    @property
    def experts_per_chip(self) -> int:
        return self.num_experts // self.chips

    @property
    def group_size(self) -> int:
        return self.num_experts // self.groups

    @property
    def sequence_tokens(self) -> int:
        return self.chunk_tokens * self.n_chunks

    def to_json(self) -> dict:
        d = asdict(self)
        d["moe_layers"] = list(self.moe_layers)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "MoeSpec":
        d = dict(d)
        d["moe_layers"] = tuple(d.get("moe_layers", ()))
        return cls(**d)

    # --- layout -------------------------------------------------------------------------------
    def identity_owners(self) -> np.ndarray:
        """owners[e] for the original layout."""
        return np.arange(self.num_experts) // self.experts_per_chip % self.rows

    def chip_of_slot(self, slot: int) -> tuple[int, int]:
        """(group, row) of a physical slot."""
        g, rest = divmod(slot, self.group_size)
        return g, rest // self.experts_per_chip

    def owners_from_physical(self, physical_to_logical) -> np.ndarray:
        owners = np.empty(self.num_experts, np.int64)
        for slot, e in enumerate(physical_to_logical):
            owners[e] = self.chip_of_slot(slot)[1]
        return owners

    def physical_from_owners(self, owners) -> list[int]:
        """Stable conversion: an expert already on its target row keeps its original slot (fewer moved
        weight shards); the others fill the free slots of their target row in ascending logical order."""
        owners = np.asarray(owners, dtype=np.int64)
        if owners.shape != (self.num_experts,):
            raise ValueError(f"owners must have {self.num_experts} entries")
        epc, gs = self.experts_per_chip, self.group_size
        phys = [None] * self.num_experts
        for g in range(self.groups):
            members = range(g * gs, (g + 1) * gs)
            for r in range(self.rows):
                want = [e for e in members if owners[e] == r]
                if len(want) != epc:
                    raise ValueError(f"group {g} row {r} gets {len(want)} experts, needs exactly {epc}")
                slots = [g * gs + r * epc + k for k in range(epc)]
                keep = [e for e in want if e in slots]
                for e in keep:
                    phys[e] = e
                free = iter(s for s in slots if s not in keep)
                for e in want:
                    if e not in keep:
                        phys[next(free)] = e
        return validate_mapping(phys, self)

    def moved_experts(self, physical_to_logical) -> int:
        return sum(1 for p, e in enumerate(physical_to_logical) if p != e)


def validate_mapping(mapping, spec: MoeSpec) -> list[int]:
    """A placement is a list of ints that keeps every expert exactly once and every expert in its group."""
    if not isinstance(mapping, list) or any(type(x) is not int for x in mapping):
        raise ValueError("Expert placement requires a list of integer expert IDs")
    if len(mapping) != spec.num_experts:
        raise ValueError(f"Expert placement requires {spec.num_experts} expert IDs, got {len(mapping)}")
    if sorted(mapping) != list(range(spec.num_experts)):
        raise ValueError("Expert placement must retain every expert exactly once")
    gs = spec.group_size
    if any(logical // gs != physical // gs for physical, logical in enumerate(mapping)):
        raise ValueError(f"Expert placement must preserve the {spec.groups} dispatch groups")
    return mapping


def inverse(mapping: list[int]) -> list[int]:
    out = [0] * len(mapping)
    for p, e in enumerate(mapping):
        out[e] = p
    return out
