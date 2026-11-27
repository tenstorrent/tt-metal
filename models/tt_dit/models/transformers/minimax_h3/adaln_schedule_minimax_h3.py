# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Resident AdaLN modulation tables for fixed sampling schedules (`MINIMAX_H3_ADALN_PRECOMPUTE`).

Every block projects `temb` into its six modulation tables on every forward: `adaln_proj` is
`Linear(2688 -> 96768)` per block, so a forward reads 26 GB of projection weights (6.5 GB per device at
TP=4) to produce a handful of table rows. A request's `temb` rows are fixed by its schedule before the
denoise loop starts, and serving repeats the same schedule request after request, so the rows are
computed once per schedule and every forward selects its step's rows from a resident table instead.

Layout. Each block keeps six `[capacity, hidden_local]` tables, one per modulation parameter. A
schedule takes `num_steps * len(roles) * MODALITY_NUM` contiguous rows, step-major, then role, then
modality: the row `_modulation_tables` puts at `slot * MODALITY_NUM + modality` for a forward whose
`temb` carries every role sits at `offset + step * rows_per_step + role * MODALITY_NUM + modality`. The
rows are made by the block's own `_modulation_tables` on device (same programs, same `temb` shape as a
full-role forward), so they already carry the `1 +` and the folded norm weight on the scale rows that
#58676 moved into the table.

Selection. A forward gets a `[len(slot_roles) * MODALITY_NUM, capacity]` bf16 one-hot selector per step
(`AdalnRowPlan.selector`) and each block gathers its step's six tables as `selector @ table` through the
same one-hot matmul as the per-token gathers (one nonzero product per output, fp32 accumulation: exact).
The result is the `[rows, hidden_local]` table `_modulation_tables` would have returned, so the
gathers, the tile-row map and the norms downstream are unchanged. A request whose slots are a subset of
the roles (t2va has no condition slot) selects only its own roles' rows.

Lifetime (#59645). A trace replay writes wherever its capture-time intermediates lived, so a long-lived
buffer allocated after any live capture can be overwritten silently. The tables are allocated once, while
no trace is live on the device (the first denoise, before warmup's captures), and never reallocated: a new
schedule's rows are written in place (`ttnn.copy`), which allocates nothing that outlives the call, and
through ops whose shapes do not depend on the schedule (`WRITE_SLOTS` `temb` slots at a time, placed by a one-hot
matmul instead of a concat), so it compiles no program while a trace is live either (a program first compiled
under a live trace can be overwritten by its replay in the same way). The other resident schedules stay as they
are; when the rows run out the least recently used ones make room.
The selector is per-forward state uploaded like `timestep`. A schedule that does not fit the capacity, or
tables that could not be allocated before a capture, leave the request on the on-device projection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from .transformer_block_minimax_h3 import MODALITY_NUM

ADALN_PRECOMPUTE_ENV = "MINIMAX_H3_ADALN_PRECOMPUTE"
ADALN_PRECOMPUTE_ROWS_ENV = "MINIMAX_H3_ADALN_PRECOMPUTE_ROWS"
# 128 rows hold a 9-step schedule with three roles (81 rows) plus warmup's 2-step one (18). A row costs
# 6 tables * hidden_local * 2 bytes per block: 128 rows are about 100 MB per device over 50 blocks at TP=4.
DEFAULT_CAPACITY_ROWS = 128
# A write projects this many `temb` slots at once: one tile row, as a forward's few slots are, so the time embedder
# and `adaln_proj` see the tile shapes of a forward and a schedule costs one projection per block, not one per step.
WRITE_SLOTS = 32


def adaln_precompute_enabled() -> bool:
    return os.environ.get(ADALN_PRECOMPUTE_ENV, "1") == "1"


def adaln_capacity_rows() -> int:
    rows = int(os.environ.get(ADALN_PRECOMPUTE_ROWS_ENV, DEFAULT_CAPACITY_ROWS))
    if rows <= 0 or rows % 32:
        raise ValueError(f"{ADALN_PRECOMPUTE_ROWS_ENV}={rows} must be a positive multiple of 32")
    return rows


@dataclass(frozen=True)
class AdalnSchedule:
    """A request's AdaLN inputs for every role: `levels[i]` is step `i`'s `[len(roles)]` float32 timestep per
    role, `endpoints[i]` its two-time endpoint (None without two-time conditioning)."""

    roles: tuple[str, ...]
    levels: tuple[torch.Tensor, ...]
    endpoints: tuple[torch.Tensor, ...] | None = None

    def __post_init__(self) -> None:
        for name, seq in (("levels", self.levels), ("endpoints", self.endpoints or ())):
            for step, level in enumerate(seq):
                if tuple(level.shape) != (len(self.roles),):
                    raise ValueError(f"{name}[{step}] has shape {tuple(level.shape)}, expected ({len(self.roles)},)")
        if self.endpoints is not None and len(self.endpoints) != len(self.levels):
            raise ValueError(f"{len(self.endpoints)} endpoints for {len(self.levels)} steps")

    @property
    def num_steps(self) -> int:
        return len(self.levels)

    def same_as(self, other: AdalnSchedule) -> bool:
        """Bitwise equal inputs: the rows built for one are exactly the other's."""

        def equal(a: tuple[torch.Tensor, ...] | None, b: tuple[torch.Tensor, ...] | None) -> bool:
            if a is None or b is None:
                return a is b
            return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))

        return self.roles == other.roles and equal(self.levels, other.levels) and equal(self.endpoints, other.endpoints)


@dataclass(frozen=True)
class AdalnSegment:
    schedule: AdalnSchedule
    offset: int

    @property
    def rows_per_step(self) -> int:
        return len(self.schedule.roles) * MODALITY_NUM

    @property
    def rows(self) -> int:
        return self.schedule.num_steps * self.rows_per_step


class AdalnRowPlan:
    """Host-side bookkeeping of the resident tables: which rows hold which schedule's steps."""

    def __init__(self, capacity: int, roles: tuple[str, ...]) -> None:
        self.capacity = capacity
        self.roles = tuple(roles)
        self.segments: list[AdalnSegment] = []

    def find(self, schedule: AdalnSchedule) -> AdalnSegment | None:
        """The resident segment of `schedule`, which becomes the most recently used (last in `segments`)."""
        for index, segment in enumerate(self.segments):
            if segment.schedule.same_as(schedule):
                self.segments.append(self.segments.pop(index))
                return segment
        return None

    def placed(self, schedule: AdalnSchedule) -> list[AdalnSegment] | None:
        """The segment list after adding `schedule`, which comes last: it takes the first free rows that hold it,
        after dropping least recently used schedules (rebuilt on their next use) until some do; None when it exceeds
        the capacity."""
        if schedule.roles != self.roles:
            raise ValueError(f"schedule roles {schedule.roles} != table roles {self.roles}")
        rows = AdalnSegment(schedule, 0).rows
        if rows > self.capacity:
            return None
        kept = list(self.segments)
        while True:
            offset = 0
            for seg in sorted(kept, key=lambda seg: seg.offset):
                if seg.offset - offset >= rows:
                    break
                offset = seg.offset + seg.rows
            if offset + rows <= self.capacity:
                return [*kept, AdalnSegment(schedule, offset)]
            kept.pop(0)

    @property
    def rows_per_step(self) -> int:
        return len(self.roles) * MODALITY_NUM

    def write_inputs(
        self, segments: list[AdalnSegment]
    ) -> list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]]:
        """`(levels, endpoints, placement, keep)` per write of the newest segment, `segments[-1]`.

        A write takes whole steps, `WRITE_SLOTS` slots at most: `levels` / `endpoints` are their `[WRITE_SLOTS]`
        values, the last one repeated as padding. `placement` is the `[capacity, WRITE_SLOTS * MODALITY_NUM]`
        one-hot that puts those steps' table rows at their place and `keep` the `[capacity, capacity]` 0/1 diagonal
        of the rows that stay as they are: the other segments' and this segment's earlier writes. Every other row
        becomes zero. The shapes are the same for any schedule and position, so a write while a trace is live
        compiles no new program.
        """
        *resident, new = segments
        schedule = new.schedule
        kept = torch.zeros(self.capacity)
        for seg in resident:
            kept[seg.offset : seg.offset + seg.rows] = 1.0
        steps = WRITE_SLOTS // len(self.roles)

        def slots(values: tuple[torch.Tensor, ...], first: int) -> torch.Tensor:
            flat = torch.cat(values[first : first + steps])
            return torch.cat([flat, flat[-1:].expand(WRITE_SLOTS - len(flat))])

        out = []
        for first in range(0, schedule.num_steps, steps):
            row = new.offset + first * self.rows_per_step
            rows = min(steps, schedule.num_steps - first) * self.rows_per_step
            placement = torch.zeros(self.capacity, WRITE_SLOTS * MODALITY_NUM)
            placement[row : row + rows, :rows] = torch.eye(rows)
            endpoints = None if schedule.endpoints is None else slots(schedule.endpoints, first)
            out.append((slots(schedule.levels, first), endpoints, placement, torch.diag(kept)))
            kept[row : row + rows] = 1.0
        return out

    def selector(self, segment: AdalnSegment, step: int, slot_roles: tuple[str, ...]) -> torch.Tensor:
        """`[len(slot_roles) * MODALITY_NUM, capacity]` one-hot: row `slot * MODALITY_NUM + modality` picks that
        slot's role's row of `step`."""
        if not 0 <= step < segment.schedule.num_steps:
            raise ValueError(f"step {step} outside the schedule's {segment.schedule.num_steps} steps")
        unknown = [role for role in slot_roles if role not in self.roles]
        if unknown:
            raise ValueError(f"slot roles {unknown} are not in the table's roles {self.roles}")
        out = torch.zeros(len(slot_roles) * MODALITY_NUM, self.capacity)
        base = segment.offset + step * segment.rows_per_step
        for slot, role in enumerate(slot_roles):
            for modality in range(MODALITY_NUM):
                out[slot * MODALITY_NUM + modality, base + self.roles.index(role) * MODALITY_NUM + modality] = 1.0
        return out
