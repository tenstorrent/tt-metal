# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Pure metadata helpers for Laguna's vLLM hybrid-KV layout.

vLLM turns Laguna's 10 full-attention and 30 sliding-window layers into four
block-table groups of ten layers.  Layers at the same ordinal in different
groups share one physical KV tensor and use disjoint block-id namespaces.  The
TT adapter needs both identities: group id selects the block table, while tensor
index selects the aliased K/V allocation.

These helpers mirror vLLM 0.24's uniform-page-size grouping algorithm without
importing vLLM internals, making the allocation contract executable and easy to
test before any device buffers or resident traces exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Sequence

LAGUNA_NUM_LAYERS = 40
LAGUNA_HYBRID_LAYER_KINDS = tuple(
    "full" if layer_index % 4 == 0 else "sliding" for layer_index in range(LAGUNA_NUM_LAYERS)
)


@dataclass(frozen=True)
class KVLayerAlias:
    """Hybrid identities for one logical decoder layer."""

    layer_index: int
    kind: str
    group_id: int
    slot_index: int

    @property
    def tensor_index(self) -> int:
        """Uniform-page layout aliases equal slots across all groups."""

        return self.slot_index


@dataclass(frozen=True)
class HybridKVLayout:
    """The ordered vLLM block-table groups and per-layer alias metadata."""

    groups: tuple[tuple[int, ...], ...]
    aliases: tuple[KVLayerAlias, ...]

    def __post_init__(self):
        if not self.groups:
            raise ValueError("hybrid KV layout must contain at least one group")
        if not self.aliases:
            raise ValueError("hybrid KV layout must contain at least one layer")
        expected = set(range(len(self.aliases)))
        actual = {layer for group in self.groups for layer in group}
        if actual != expected:
            raise ValueError(f"hybrid KV groups cover layers {sorted(actual)}, expected {sorted(expected)}")
        if sum(len(group) for group in self.groups) != len(self.aliases):
            raise ValueError("hybrid KV groups contain a duplicate logical layer")

    @property
    def num_groups(self) -> int:
        return len(self.groups)

    @property
    def num_tensors(self) -> int:
        return max(alias.tensor_index for alias in self.aliases) + 1

    @property
    def representative_layers(self) -> tuple[int, ...]:
        return tuple(group[0] for group in self.groups)

    def expand_group_values(self, values: Sequence[Any]) -> list[Any]:
        """Expand one page-table buffer per group into logical layer order."""

        if len(values) != self.num_groups:
            raise ValueError(f"got {len(values)} group values for {self.num_groups} groups")
        return [values[alias.group_id] for alias in self.aliases]


def build_hybrid_kv_layout(layer_kinds: Sequence[str]) -> HybridKVLayout:
    """Mirror vLLM's uniform-page-size grouping for an ordered layer pattern.

    Layers are first grouped by attention kind in first-seen order.  The common
    group size is the smallest kind count, except vLLM's 1.5x heuristic pads to
    the largest count when counts are close.  A kind requiring multiple groups
    is split with ``layers[i::num_groups]`` so pipeline stages retain the same
    repeated pattern.
    """

    kinds = tuple(str(kind) for kind in layer_kinds)
    if not kinds:
        raise ValueError("hybrid KV grouping requires at least one layer kind")
    if any(not kind for kind in kinds):
        raise ValueError("hybrid KV layer kinds must be non-empty strings")

    same_kind: dict[str, list[int]] = {}
    for layer_index, kind in enumerate(kinds):
        same_kind.setdefault(kind, []).append(layer_index)

    counts = [len(layers) for layers in same_kind.values()]
    min_count, max_count = min(counts), max(counts)
    group_size = max_count if max_count < min_count * 1.5 else min_count

    groups: list[tuple[int, ...]] = []
    group_kinds: list[str] = []
    for kind, layers in same_kind.items():
        num_groups = ceil(len(layers) / group_size)
        for group_offset in range(num_groups):
            group = tuple(layers[group_offset::num_groups])
            if group:
                groups.append(group)
                group_kinds.append(kind)

    aliases: list[KVLayerAlias | None] = [None] * len(kinds)
    for group_id, (kind, group) in enumerate(zip(group_kinds, groups, strict=True)):
        for slot_index, layer_index in enumerate(group):
            aliases[layer_index] = KVLayerAlias(layer_index, kind, group_id, slot_index)
    if any(alias is None for alias in aliases):
        raise ValueError("hybrid KV grouping failed to assign every layer")
    return HybridKVLayout(tuple(groups), tuple(aliases))  # type: ignore[arg-type]


def build_laguna_hybrid_kv_layout(layer_kinds: Sequence[str]) -> HybridKVLayout:
    """Build Laguna's production layout and reject reduced or reordered stacks.

    Tensor aliasing is a whole-model vLLM contract: the four groups share ten
    physical buffers by slot.  Applying that contract to a reduced layer bring-up,
    or to a checkpoint whose attention pattern changed, can make unrelated logical
    layers overwrite one another.  Hybrid allocation therefore requires the exact
    published 40-layer ``full, sliding, sliding, sliding`` repetition.
    """

    kinds = tuple(str(kind) for kind in layer_kinds)
    if kinds != LAGUNA_HYBRID_LAYER_KINDS:
        raise ValueError(
            "Laguna hybrid KV requires the exact 40-layer full/sliding pattern; " f"got {len(kinds)} layer kinds"
        )
    layout = build_hybrid_kv_layout(kinds)
    expected_groups = tuple(tuple(range(group_id, LAGUNA_NUM_LAYERS, 4)) for group_id in range(4))
    if layout.groups != expected_groups or layout.num_tensors != 10:
        raise ValueError("Laguna hybrid KV grouping drifted from four groups sharing ten physical tensors")
    return layout


def validate_per_layer_tensor_aliases(
    per_layer_specs: Sequence[tuple[Sequence[int], Any, int]],
    layout: HybridKVLayout,
) -> dict[int, tuple[tuple[int, ...], Any]]:
    """Validate plugin ``(shape, dtype, tensor_idx)`` metadata before allocation.

    Returns one ``(shape, dtype)`` descriptor per unique physical tensor.  Every
    logical layer still receives its own cache dictionary later, but dictionaries
    with the same tensor index must point at the same K/V objects.
    """

    if len(per_layer_specs) != len(layout.aliases):
        raise ValueError(f"got {len(per_layer_specs)} per-layer KV specs for {len(layout.aliases)} logical layers")
    unique: dict[int, tuple[tuple[int, ...], Any]] = {}
    for alias, entry in zip(layout.aliases, per_layer_specs, strict=True):
        if len(entry) != 3:
            raise ValueError(f"layer {alias.layer_index} KV spec must be (shape, dtype, tensor_idx)")
        shape, dtype, tensor_index = entry
        tensor_index = int(tensor_index)
        if tensor_index != alias.tensor_index:
            raise ValueError(
                f"layer {alias.layer_index} plugin tensor_idx={tensor_index}, expected "
                f"slot-derived tensor_idx={alias.tensor_index}"
            )
        descriptor = (tuple(int(dim) for dim in shape), dtype)
        prior = unique.setdefault(tensor_index, descriptor)
        if prior != descriptor:
            raise ValueError(f"physical KV tensor {tensor_index} has inconsistent descriptors {prior} and {descriptor}")
    if set(unique) != set(range(layout.num_tensors)):
        raise ValueError(f"physical KV tensor indices {sorted(unique)} are not contiguous")
    return unique
