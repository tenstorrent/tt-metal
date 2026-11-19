# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The media embedding rows a packed forward writes over its image positions, laid out like its tokens."""

import math
from dataclasses import dataclass
from typing import Callable, NamedTuple

import numpy as np
import torch

import ttnn
from models.demos.common.prefill.media_stub import stub_media_key
from models.demos.common.prefill.prefill_chunk import MediaRef, PrefillChunk


def _positions(chunk_start: int, sp: int, chunk_local: int) -> torch.Tensor:
    """rotated_chunk_positions(chunk_start, sp, chunk_local) as an [sp, chunk_local] tensor."""
    chunk_global = sp * chunk_local
    chip = torch.arange(sp).unsqueeze(1)
    boundary = (chunk_start // chunk_local) % sp
    first = (chunk_start // chunk_global + (chip < boundary)) * chunk_local + (chip == boundary) * (
        chunk_start % chunk_local
    )
    local_row = first + torch.arange(chunk_local)
    return (local_row // chunk_local) * chunk_global + chip * chunk_local + local_row % chunk_local


def next_table_height(height: int) -> int:
    return math.ceil(int(height * 1.25) / ttnn.TILE_SIZE) * ttnn.TILE_SIZE


def table_height(rows: int) -> int:
    """Snap a media table's height up a ~1.25x ladder of whole tiles: 32, 64, 96, 128, 160, 224, ...

    k moves with how many images a forward names, so an exact height would compile a new program for
    every image count and could never be preallocated. Snapping leaves a few shapes the program cache hits
    and compile() can warm. From 97 rows up, padding stays under ~30% of the table; smaller tables are whole tiles."""
    height = ttnn.TILE_SIZE
    while height < rows:
        height = next_table_height(height)
    return height


def warm_media_refs(sp: int, chunk: int, n_chunks: int, by_position: bool) -> list[tuple[int, list[MediaRef]]]:
    """(height, refs) for every table height a forward of n_chunks chunks can reach. Each height gets the
    smallest image token count that lands in it, as stub-keyed refs filling chunks 0, 1, ... in turn. by_position
    fills only the first chip's positions of each chunk, the most one SP row can be given."""
    per_chunk = chunk // sp if by_position else chunk
    height, below, plan = ttnn.TILE_SIZE, 0, []
    while (n := max(1, below if by_position else sp * below)) <= n_chunks * per_chunk:
        sizes = [min(per_chunk, n - k * per_chunk) for k in range(math.ceil(n / per_chunk))]
        plan.append((height, [MediaRef(k, 0, m, stub_media_key(m, k)) for k, m in enumerate(sizes)]))
        height, below = next_table_height(height), height
    return plan


class Run(NamedTuple):
    sp_row: int
    slot: int  # first table slot; slot 0 is the zero row
    rows: np.ndarray  # consecutive media rows, a view into the fetched value


@dataclass(frozen=True)
class SplicePlan:
    height: int
    lookup: torch.Tensor  # [sp, s_total] (2D) or [sp, s_local] (1D) int32, uploaded as uint32; 0 = keep the token
    is_img: torch.Tensor  # [s_total] bool
    runs: list[Run]


def plan_media_splice(
    refs: list[MediaRef],
    chunks: list[PrefillChunk],
    rows_of: Callable[[MediaRef], np.ndarray],
    sp: int,
    chunk_local: int,
    by_position: bool = False,
) -> SplicePlan:
    """Where each image row of a forward with refs goes. height is table_height(k + 1), so a forward reuses one of
    a few table shapes. Each run puts consecutive media rows viewed from rows_of at table slots slot.. of its SP
    row; the caller copies them straight into the table, so the media rows are copied once. Row 0 of every table
    is zeros and lookup 0 means "not this row's", so a row contributes nothing where it was not assigned.

    The 2D embedding (vocab sharded over SP) splices before its reduce-scatter. by_position is for the 1D
    embedding, which has no reduce-scatter: each image row goes to the SP row whose chips hold its position, and
    lookup is [sp, s_local], over that row's own positions.

    Positions are in the forward's chip-major token order (_positions), not natural order: ``is_img`` and the 2D
    ``lookup`` follow the all-gathered tokens _forward_2d looks up, and row r of the 1D ``lookup`` follows SP row r's
    own tokens.
    """
    positions = torch.cat([_positions(c.actual_start, sp, chunk_local) for c in chunks], dim=1)
    chunk_of = torch.arange(len(chunks)).repeat_interleave(chunk_local).expand(sp, -1)
    flat_pos = positions.flatten()

    flats, ref_id, row_in_ref = [], [], []
    for j, ref in enumerate(refs):
        c = chunks[ref.chunk_index]
        lo, hi = max(ref.start_pos, c.actual_start), min(ref.end_pos, c.actual_end)
        sel = ((chunk_of == ref.chunk_index) & (positions >= lo) & (positions < hi)).flatten().nonzero().flatten()
        flats.append(sel)
        ref_id.append(torch.full_like(sel, j))
        row_in_ref.append(flat_pos[sel] - ref.start_pos)
    flats, ref_id, row_in_ref = torch.cat(flats), torch.cat(ref_id), torch.cat(row_in_ref)

    # Slot 0 of every row stays zero. Each SP row's slots are 1..count in order, so consecutive media rows fill
    # consecutive slots and a run is one copy rather than a strided scatter.
    order = torch.arange(flats.numel())
    s_total = positions.numel()
    if by_position:
        # Each image row goes to the SP row holding its position.
        sp_row, by_row = (flats // (s_total // sp)).sort(stable=True)
        flats, ref_id, row_in_ref = flats[by_row], ref_id[by_row], row_in_ref[by_row]
        counts = torch.bincount(sp_row, minlength=sp)
        slot = order - (counts.cumsum(0) - counts)[sp_row] + 1
    else:
        # Each image row goes to exactly one SP row, the n-th to row n // per_row, so the reduce-scatter that
        # already runs sums it into place: adding the other rows' zeros is exact in bf16.
        per_row = max(1, math.ceil(flats.numel() / sp))
        sp_row, slot = order // per_row, order % per_row + 1

    height = table_height(int(slot.max()) + 1 if slot.numel() else 1)
    lookup = torch.zeros(sp, s_total // sp if by_position else s_total, dtype=torch.int32)
    is_img = torch.zeros(s_total, dtype=torch.bool)
    if by_position:
        lookup.view(-1)[flats] = slot.to(torch.int32)
    else:
        lookup[sp_row, flats] = slot.to(torch.int32)
    is_img[flats] = True

    # A run breaks where the SP row or the ref changes, or the media rows stop being consecutive.
    brk = torch.ones(flats.numel(), dtype=torch.bool)
    brk[1:] = (ref_id[1:] != ref_id[:-1]) | (sp_row[1:] != sp_row[:-1]) | (row_in_ref[1:] != row_in_ref[:-1] + 1)
    starts = brk.nonzero().flatten().tolist()
    runs = []
    for a, b in zip(starts, starts[1:] + [flats.numel()]):
        first = int(row_in_ref[a])
        runs.append(Run(int(sp_row[a]), int(slot[a]), rows_of(refs[int(ref_id[a])])[first : first + b - a]))
    return SplicePlan(height, lookup, is_img, runs)
