# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The media embedding rows a packed forward writes over its image positions, laid out like its tokens."""

from typing import Callable, Optional

import numpy as np
import torch

from models.demos.common.prefill.chunk_layout import rotated_chunk_positions
from models.demos.common.prefill.prefill_chunk import MediaRef, PrefillChunk


def build_injection(
    refs: list[MediaRef],
    chunks: list[PrefillChunk],
    rows_of: Callable[[MediaRef], np.ndarray],
    sp: int,
    chunk_local: int,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """(rows [sp, F * chunk_local, hidden] bf16, mask [sp, F * chunk_local]) in the forward's per-chip order
    (pack_forward_tokens), or None without refs. rows_of(ref) is the ref's [num_tokens, hidden] embeddings as
    bf16 bit patterns (uint16); only the rows inside each ref's chunk are written."""
    if not refs:
        return None
    per_chunk = [rotated_chunk_positions(c.actual_start, sp, chunk_local) for c in chunks]
    positions = torch.tensor([[p for chunk in per_chunk for p in chunk[c]] for c in range(sp)])
    chunk_of = torch.arange(len(chunks)).repeat_interleave(chunk_local).expand(sp, -1)
    rows, mask = None, torch.zeros(positions.shape, dtype=torch.bool)
    for ref in refs:
        c = chunks[ref.chunk_index]
        lo, hi = max(ref.start_pos, c.actual_start), min(ref.end_pos, c.actual_end)
        sel = (chunk_of == ref.chunk_index) & (positions >= lo) & (positions < hi)
        emb = rows_of(ref)
        assert emb.shape[0] == ref.num_tokens, f"{emb.shape[0]} rows for {ref}"
        if rows is None:
            rows = torch.zeros(*positions.shape, emb.shape[1], dtype=torch.bfloat16)
        picked = np.ascontiguousarray(emb[(positions[sel] - ref.start_pos).numpy()])
        rows[sel] = torch.from_numpy(picked).view(torch.bfloat16)
        mask |= sel
    return rows, mask
