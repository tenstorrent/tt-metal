# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Does a WIDER chunk page table corrupt already-committed KV, or only future positions?

This decides whether the DFlash verify forward can be traced. ``_forward_prefill_chunk_masked_tp``
builds its chunk page table with a width derived from ``valid_len``::

    blk0 = chunk_start // block_size
    blkN = num_blocks_in_seq(chunk_start + valid_len, block_size)   # <- valid_len
    chunk_pt = page_table[:, blk0:blkN]

and carries the comment "Fill K/V only for real blocks (ceil(valid_len/64)); padded writes would
corrupt block 0". A trace cannot have a shape that depends on ``valid_len``, so the fix would be a
FIXED width -- 3 blocks always covers a 128-row bucket at any alignment -- with the contents staged
per step. That is only safe if the extra written rows land on **future** positions, which
speculation rewrites anyway, rather than on **committed** ones.

Reasoning does not settle it and the failure mode is silent KV corruption, so measure it. The op is
all that is needed; no 27B, no model.

MEASURED (blk0 = 0, BUCKET = 128, BLOCK = 64, so S = 128):

    width=1 (page_len= 64 < S): wrote positions [0..63]   64 rows, values correct
    width=2 (page_len=128 = S): wrote positions [0..127] 128 rows, values correct
    width=3 (page_len=192 > S): wrote positions [0..127] 128 rows, values correct

**A wider page table is safe.** ``paged_fill_cache`` writes exactly ``min(S, page_len)`` rows, each
at its correct sequential position; surplus pages are left untouched (page 2 still held the
sentinel in the width-3 case). There is no wraparound and no write below the range, so the
"padded writes would corrupt block 0" warning in ``model.py`` does not describe this shape -- it
must be about the batched path, where users share pages.

CONSEQUENCE FOR TRACING: the chunk page table can be given the FIXED width ``bucket // block_size``
(= 2 here) with its contents staged per step, which removes the last shape-varying input from the
verify forward. The rows past ``valid_len`` receive padding K/V, but they are FUTURE positions --
the next speculative step rewrites from the accepted index forward, and attention never reads past
the committed length.

CAVEAT: this was measured with ``blk0 = 0`` and therefore a block-aligned ``chunk_start``. The fill's
row 0 maps to the first LISTED page's row 0, so a ``chunk_start`` that is not a multiple of
``block_size`` would be written ``chunk_start % block_size`` rows early. The DFlash target advances
its anchor by whole buckets, so it is always block-aligned -- but anything that breaks that
invariant breaks this result.

WHAT IT CHECKS
--------------
Fill a paged cache with a sentinel, write ``S`` known rows through page tables of increasing width,
read back, and report exactly which rows changed. ``attention/tp.py`` slices the fill down to
``page_len`` when ``page_len < S``, so this mirrors that too.

Run::

    MESH_DEVICE=N150 pytest -svq models/demos/blackhole/qwen36/tests/unit/test_paged_fill_cache_width.py
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn

BLOCK = 64  # PAGED_BLOCK_SIZE in the DFlash target
NUM_BLOCKS = 8
NKV = 2
HD = 64
BUCKET = 128  # the verify forward's bucket
VALID = 16  # a speculative block


def _mesh_shape():
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    return {"P150": (1, 1), "N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(name, (1, 1))


MESH_SHAPE = _mesh_shape()
_MULTI = MESH_SHAPE != (1, 1)


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, **({"fabric_config": ttnn.FabricConfig.FABRIC_1D} if _MULTI else {})}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_paged_fill_cache_width(mesh_device, device_params):
    """Report which cache rows a width-N chunk page table actually writes."""
    del device_params
    mapper = {"mesh_mapper": ttnn.ReplicateTensorToMesh(mesh_device)} if _MULTI else {}
    composer = {"mesh_composer": ttnn.ConcatMeshToTensor(mesh_device, dim=0)} if _MULTI else {}

    # Row r of the fill carries the value r+1, so a read-back tells us exactly where each row went.
    fill_t = torch.arange(1, BUCKET + 1, dtype=torch.float32).reshape(1, 1, BUCKET, 1)
    fill_t = fill_t.expand(1, NKV, BUCKET, HD).contiguous().to(torch.bfloat16)

    for width in (1, 2, 3):
        # Fresh sentinel cache each time: -1 everywhere means "never written".
        cache_t = torch.full((NUM_BLOCKS, NKV, BLOCK, HD), -1.0, dtype=torch.bfloat16)
        cache = ttnn.from_torch(cache_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, **mapper)
        fill = ttnn.from_torch(fill_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, **mapper)

        # blk0 = 0 (chunk_start = 0). Width 1 is what the code does today for valid_len=16;
        # width 3 is what a fixed-shape (trace-able) page table would use.
        pt = torch.arange(width, dtype=torch.int32).reshape(1, width)
        pt_tt = ttnn.from_torch(pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, **mapper)

        page_len = width * BLOCK
        src = fill
        if page_len < BUCKET:  # mirrors attention/tp.py
            src = ttnn.slice(fill, (0, 0, 0, 0), (1, NKV, page_len, HD))

        try:
            ttnn.experimental.paged_fill_cache(cache, src, pt_tt, batch_idx=0)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"width={width} UNSUPPORTED: {type(e).__name__}: {str(e).splitlines()[0][:150]}")
            continue

        got = ttnn.to_torch(ttnn.get_device_tensors(cache)[0] if _MULTI else cache).float()
        # Flatten pages back to sequence order: block b row r -> position b*BLOCK + r.
        seq = got.permute(0, 2, 1, 3).reshape(NUM_BLOCKS * BLOCK, NKV, HD)[:, 0, 0]
        written = (seq != -1).nonzero().flatten().tolist()
        lo, hi = (written[0], written[-1]) if written else (None, None)
        # Is every written position carrying the value the fill intended for it?
        placed_ok = all(abs(seq[p].item() - (p + 1)) < 0.5 for p in written) if written else None
        logger.info(
            f"width={width} (page_len={page_len}): wrote positions [{lo}..{hi}] "
            f"({len(written)} rows), value-correct={placed_ok}"
        )
        print(f">>> width={width} wrote=[{lo}..{hi}] n={len(written)} correct={placed_ok}")

        for t in (cache, fill, pt_tt):
            ttnn.deallocate(t)

    logger.info(
        f"VERDICT: a fixed width-3 table is safe for tracing iff every write lands at position "
        f">= {VALID} only in the FUTURE (positions {VALID}..) and never below it."
    )
