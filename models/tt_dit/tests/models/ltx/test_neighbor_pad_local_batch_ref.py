# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU check of neighbor_pad_async's batched local copy (TT_NEIGHBOR_PAD_LOCAL_BATCH=<n>).

Emulates local_copy_reader/local_copy_writer at stick granularity through a CB ring of 2*batch pages,
with the host's batch choice (largest divisor of the row length <= n that fits 32 KB). The batched
copy must write the same bytes to the same output sticks as the per-stick copy, and no batch may
wrap the CB (a wrapped batch would read or write past the CB end on device).
Run: python -m pytest --noconftest <this file>
"""

import pytest
import torch

MAX_BATCH_BYTES = 32 * 1024


def _pick_batch(env, row_sticks, page_size):
    # Mirrors the program factory.
    if env is None:
        return 1
    cap = min(int(env), MAX_BATCH_BYTES // page_size)
    for b in range(min(cap, row_sticks), 1, -1):
        if row_sticks % b == 0:
            return b
    return 1


def _emulate(inp, rows, row_sticks, out_row_sticks, stick_start, padding_left, t_front_off, masked_rows, valid, batch):
    """inp: [rows*row_sticks, C]. Returns the output sticks written by the local copy (others NaN)."""
    ch = inp.shape[1]
    out = torch.full(((rows + 2 * padding_left) * out_row_sticks + t_front_off, ch), float("nan"))
    cb = torch.full((2 * batch, ch), float("nan"))
    wr = rd = 0  # CB write/read page pointers
    for t in range(rows):
        src = t * row_sticks
        dst = (t + padding_left) * out_row_sticks + stick_start + t_front_off
        for it in range(0, row_sticks, batch):
            assert wr + batch <= 2 * batch, "batch wraps the CB"
            for j in range(batch):
                cb[wr + j] = inp[src]
                src += 1
            wr = (wr + batch) % (2 * batch)
            assert rd + batch <= 2 * batch, "batch wraps the CB"
            for j in range(batch):
                out[dst] = 0 if (t in masked_rows or it + j >= valid) else cb[rd + j]
                dst += 1
            rd = (rd + batch) % (2 * batch)
    return out


# (rows, row_sticks, C, pad_w, env): LTX decode shard shapes (544x960/145f on 2x4) plus odd row lengths.
CASES = [
    (72, 64, 128, 1, "128"),
    (36, 32, 256, 1, "128"),
    (36, 32, 512, 1, "128"),
    (18, 16, 1024, 1, "128"),
    (9, 60, 128, 1, "64"),
    (5, 7, 128, 0, "64"),  # prime row length
    (4, 64, 128, 1, "1"),
    (4, 64, 128, 1, None),
]


@pytest.mark.parametrize("rows,row_sticks,ch,pad_w,env", CASES)
@pytest.mark.parametrize("masked", [False, True])
def test_local_batch_matches_per_stick(rows, row_sticks, ch, pad_w, env, masked):
    torch.manual_seed(0)
    page_size = ch * 2
    batch = _pick_batch(env, row_sticks, page_size)
    assert row_sticks % batch == 0 and 2 * batch * page_size <= 2 * MAX_BATCH_BYTES
    inp = torch.randn(rows * row_sticks, 4)
    kw = dict(
        rows=rows,
        row_sticks=row_sticks,
        out_row_sticks=row_sticks + 2 * pad_w,
        stick_start=pad_w,
        padding_left=1,
        t_front_off=0,
        masked_rows={rows - 1} if masked else set(),
        valid=row_sticks - 3 if masked else row_sticks,
    )
    ref = _emulate(inp, batch=1, **kw)
    got = _emulate(inp, batch=batch, **kw)
    assert torch.equal(ref.isnan(), got.isnan())
    assert torch.equal(ref.nan_to_num(), got.nan_to_num())


def test_pick_batch_ltx_shapes():
    assert _pick_batch("128", 64, 256) == 64
    assert _pick_batch("128", 32, 512) == 32
    assert _pick_batch("128", 32, 1024) == 32
    assert _pick_batch("128", 16, 2048) == 16
    assert _pick_batch("128", 60, 256) == 60
    assert _pick_batch("16", 60, 256) == 15
    assert _pick_batch("128", 7, 256) == 7
    assert _pick_batch("0", 64, 256) == 1
    assert _pick_batch(None, 64, 256) == 1
