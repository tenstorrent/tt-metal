# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The media embedding rows a packed forward writes over its image positions, laid out like its tokens."""

from typing import Callable, Optional

import numpy as np
import torch

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


def build_injection(
    refs: list[MediaRef],
    chunks: list[PrefillChunk],
    rows_of: Callable[[MediaRef], np.ndarray],
    sp: int,
    chunk_local: int,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """(rows [sp, F * chunk_local, hidden] bf16, mask [sp, F * chunk_local]) in the forward's per-chip order
    (pack_forward_tokens), or None without refs. rows_of(ref) is the ref's [num_tokens, hidden] embeddings as
    bf16 bit patterns (uint16); only the rows inside each ref's chunk are written. Rows outside the mask are
    uninitialised: inject_media_embeddings selects on the mask and never reads them."""
    if not refs:
        return None
    positions = torch.cat([_positions(c.actual_start, sp, chunk_local) for c in chunks], dim=1)
    chunk_of = torch.arange(len(chunks)).repeat_interleave(chunk_local).expand(sp, -1)
    rows, mask = None, torch.zeros(positions.shape, dtype=torch.bool)
    for ref in refs:
        c = chunks[ref.chunk_index]
        lo, hi = max(ref.start_pos, c.actual_start), min(ref.end_pos, c.actual_end)
        sel = (chunk_of == ref.chunk_index) & (positions >= lo) & (positions < hi)
        emb = rows_of(ref)
        assert emb.shape[0] == ref.num_tokens, f"{emb.shape[0]} rows for {ref}"
        if rows is None:
            rows = torch.empty(*positions.shape, emb.shape[1], dtype=torch.bfloat16)
        picked = np.ascontiguousarray(emb[(positions[sel] - ref.start_pos).numpy()])
        rows[sel] = torch.from_numpy(picked).view(torch.bfloat16)
        mask |= sel
    return rows, mask


def table_height(rows: int) -> int:
    """Snap a partial table's height to a power of two (min 32, one tile).

    k moves with how many images a forward names, so an exact height would compile a new program for
    every image count and could never be preallocated. Snapping leaves a handful of shapes: the program
    cache hits, and compile() can warm each one so trace capture has somewhere to put them."""
    height = 32
    while height < rows:
        height *= 2
    return height


def build_media_partials(
    refs: list[MediaRef],
    chunks: list[PrefillChunk],
    rows_of: Callable[[MediaRef], np.ndarray],
    sp: int,
    chunk_local: int,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """(table [sp, h, hidden] bf16, lookup [sp, s_total] uint32, is_img [s_total] bool), or None
    without refs. h is table_height(k + 1): a power of two, so a forward reuses one of a few shapes.

    For the 2D embedding's pre-reduce-scatter partials. Each image row is assigned to exactly one SP
    row, round-robin, so the reduce-scatter that already runs sums it into place: adding zeros is exact
    in bf16. Row 0 of every table is zeros and lookup 0 means "not this row's", so a row contributes
    nothing where it was not assigned.

    ``lookup`` and ``is_img`` are indexed by the all-gathered token order (chip-major, _positions), not
    by natural position: that is the order the lookup sees inside _forward_2d.
    """
    if not refs:
        return None
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

    # Round-robin: the n-th image row goes to SP row n % sp, at slot n // sp + 1. Slot 0 stays zero, so
    # a row that was not assigned a position contributes nothing to the reduce-scatter's sum.
    order = torch.arange(flats.numel())
    sp_row, slot = order % sp, order // sp + 1

    s_total = positions.numel()
    hidden = rows_of(refs[0]).shape[1]
    table = torch.zeros(sp, table_height(int(slot.max()) + 1 if slot.numel() else 1), hidden, dtype=torch.bfloat16)
    lookup = torch.zeros(sp, s_total, dtype=torch.int32)
    is_img = torch.zeros(s_total, dtype=torch.bool)
    lookup[sp_row, flats] = slot.to(torch.int32)
    is_img[flats] = True
    for j, ref in enumerate(refs):  # one gather per ref, not one per image row
        m = ref_id == j
        picked = np.ascontiguousarray(rows_of(ref)[row_in_ref[m].numpy()])
        table[sp_row[m], slot[m]] = torch.from_numpy(picked).view(torch.bfloat16)
    return table, lookup, is_img
