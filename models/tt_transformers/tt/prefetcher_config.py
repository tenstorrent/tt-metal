# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Configuration loading and pure receiver placement for tt-transformers prefetchers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import yaml

_CONFIG_PATH = Path(__file__).parent / "prefetcher/prefetcher_config.yaml"
with open(_CONFIG_PATH, encoding="utf-8") as config_file:
    ARCH_CONFIG = yaml.safe_load(config_file)


@dataclass(frozen=True)
class TensorPrefetcherReceiverLayout:
    """A bank-major logical receiver ring and its abstract program grid."""

    receiver_coords: Tuple[Tuple[int, int], ...]
    ring_cols: int
    ring_rows: int
    profile_name: str
    used_fallback: bool
    left_columns: int
    ambiguous_columns: int
    right_columns: int


def _materialize_profile(profile_set: Mapping, receivers_per_bank: int) -> Optional[Mapping]:
    """Select an ordered per-bank subset from one maximal receiver placement."""
    selection = profile_set.get("selections", {}).get(receivers_per_bank)
    banks = profile_set.get("banks")
    if selection is None or banks is None:
        return None

    bank_slot_indices = selection.get("bank_slot_indices")
    if bank_slot_indices is None or len(bank_slot_indices) != len(banks):
        return None

    selected_banks = []
    for bank_slots, selected_indices in zip(banks, bank_slot_indices):
        if len(selected_indices) != receivers_per_bank or len(set(selected_indices)) != receivers_per_bank:
            return None
        if any(not 0 <= index < len(bank_slots) for index in selected_indices):
            return None
        selected_banks.append([bank_slots[index] for index in selected_indices])
    return {"name": selection["name"], "banks": selected_banks}


def _profile_for(
    tensor_config: Mapping, num_dram_banks: int, receivers_per_bank: int
) -> Tuple[Optional[Mapping], Optional[int]]:
    profiles = tensor_config["profiles"]
    exact_profile = _materialize_profile(profiles.get(num_dram_banks, {}), receivers_per_bank)
    if exact_profile is not None:
        return exact_profile, num_dram_banks

    fallback_bank_count = tensor_config["fallback_profile_dram_banks"]
    return _materialize_profile(profiles.get(fallback_bank_count, {}), receivers_per_bank), fallback_bank_count


def _source_bank_indices(num_dram_banks: int, profile_bank_count: int) -> Tuple[int, ...]:
    """Map runtime bank IDs to same-side banks in the configured profile."""
    runtime_left_banks = num_dram_banks // 2
    profile_left_banks = profile_bank_count // 2
    profile_right_banks = profile_bank_count - profile_left_banks
    if profile_left_banks == 0 or profile_right_banks == 0:
        return ()

    source_indices = []
    for bank_idx in range(num_dram_banks):
        if bank_idx < runtime_left_banks:
            source_indices.append(bank_idx % profile_left_banks)
        else:
            right_idx = bank_idx - runtime_left_banks
            source_indices.append(profile_left_banks + right_idx % profile_right_banks)
    return tuple(source_indices)


def allocate_tensor_prefetcher_receiver_layout(
    compute_grid: Tuple[int, int],
    dram_worker_anchors: Sequence[Tuple[int, int]],
    num_dram_banks: int,
    receivers_per_bank: int,
    tensor_config: Optional[Mapping] = None,
    *,
    device_right_starts: Optional[Sequence[int]] = None,
) -> Optional[TensorPrefetcherReceiverLayout]:
    """Materialize a configured side-relative profile on a harvested logical worker grid.

    The optimal DRAM-worker mapping contributes only its X coordinates. The first
    ``num_dram_banks // 2`` banks are adjacent to the left DRAM column and the remainder
    are adjacent to the middle DRAM column. The latter group's logical X is the first
    right-side worker column. When devices have different splits, columns classified on
    the same side by every device are used for exact placement. Columns between the
    splits remain available between preferred- and opposite-side fallback.
    """
    tensor_config = tensor_config or ARCH_CONFIG["blackhole"]["tensor_prefetcher"]
    grid_x, grid_y = compute_grid
    if grid_x <= 1 or grid_y <= 0 or num_dram_banks <= 1 or receivers_per_bank <= 0:
        return None
    if len(dram_worker_anchors) != num_dram_banks:
        return None

    runtime_left_banks = num_dram_banks // 2
    right_anchors = dram_worker_anchors[runtime_left_banks:]
    if not right_anchors:
        return None
    local_right_start = min(anchor[0] for anchor in right_anchors)
    right_starts = tuple(int(start) for start in (device_right_starts or (local_right_start,)))
    if not right_starts or any(not 0 < start < grid_x for start in right_starts):
        return None
    left_end = min(right_starts)
    right_start = max(right_starts)

    profile, profile_bank_count = _profile_for(tensor_config, num_dram_banks, receivers_per_bank)
    if profile is None or profile_bank_count is None:
        return None

    source_bank_indices = _source_bank_indices(num_dram_banks, profile_bank_count)
    if len(source_bank_indices) != num_dram_banks:
        return None
    profile_banks = profile["banks"]
    if len(profile_banks) != profile_bank_count:
        return None

    slots = []
    for source_bank_idx in source_bank_indices:
        bank_slots = profile_banks[source_bank_idx]
        if len(bank_slots) != receivers_per_bank:
            return None
        for slot in bank_slots:
            if len(slot) != 3 or slot[0] not in ("left", "right"):
                return None
            slots.append((slot[0], int(slot[1]), int(slot[2])))

    side_columns = {
        "left": tuple(range(left_end)),
        "right": tuple(range(right_start, grid_x)),
    }
    ambiguous_columns = tuple(range(left_end, right_start))
    allocated = [None] * len(slots)
    used = set()
    used_fallback = profile_bank_count != num_dram_banks or left_end != right_start

    def exact_coord(slot):
        side, column_ordinal, row = slot
        columns = side_columns[side]
        if not 0 <= column_ordinal < len(columns) or not 0 <= row < grid_y:
            return None
        return columns[column_ordinal], row

    # Reserve every valid configured slot before fallback placement. This keeps a
    # missing harvested-column slot from stealing a later slot's exact core.
    for ring_pos, slot in enumerate(slots):
        coord = exact_coord(slot)
        if coord is not None and coord not in used:
            allocated[ring_pos] = coord
            used.add(coord)
        else:
            used_fallback = True

    spill_rows = tuple(dict.fromkeys(int(row) for row in tensor_config["fallback"]["spill_rows"]))
    side_order = tuple(tensor_config["fallback"]["side_order"])
    if side_order != ("preferred", "opposite"):
        return None

    for ring_pos, slot in enumerate(slots):
        if allocated[ring_pos] is not None:
            continue

        preferred_side, preferred_ordinal, preferred_row = slot
        opposite_side = "right" if preferred_side == "left" else "left"
        candidate_rows = tuple(row for row in dict.fromkeys((preferred_row, *spill_rows)) if 0 <= row < grid_y)

        replacement = None
        candidate_column_groups = (
            (preferred_side, side_columns[preferred_side]),
            ("ambiguous", ambiguous_columns),
            (opposite_side, side_columns[opposite_side]),
        )
        for side, columns in candidate_column_groups:
            if not columns:
                continue
            if side == "ambiguous":
                column_order = columns if preferred_side == "left" else tuple(reversed(columns))
            else:
                ordinal_order = sorted(
                    range(len(columns)),
                    key=lambda ordinal: (abs(ordinal - min(preferred_ordinal, len(columns) - 1)), ordinal),
                )
                column_order = tuple(columns[ordinal] for ordinal in ordinal_order)
            for row in candidate_rows:
                for column in column_order:
                    coord = (column, row)
                    if coord not in used:
                        replacement = coord
                        break
                if replacement is not None:
                    break
            if replacement is not None:
                break

        if replacement is None:
            return None
        allocated[ring_pos] = replacement
        used.add(replacement)

    if len(used) != num_dram_banks * receivers_per_bank:
        return None

    ring_grid = tensor_config["ring_grid"]
    if ring_grid != {"columns": "dram_banks", "rows": "receivers_per_bank"}:
        return None

    return TensorPrefetcherReceiverLayout(
        receiver_coords=tuple(allocated),
        ring_cols=num_dram_banks,
        ring_rows=receivers_per_bank,
        profile_name=profile["name"],
        used_fallback=used_fallback,
        left_columns=left_end,
        ambiguous_columns=right_start - left_end,
        right_columns=grid_x - right_start,
    )
