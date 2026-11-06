# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Device tests for two optional indexer_score_dsa knobs on the Hi=1 wide lanes form (QSA decode lanes):

  KC  : program_config=IndexerScoreProgramConfig(k_chunk_size=32*KC), KC in {1,2,4,8} (qsa env QWEN38_IDX_KC)
  ROW : consumed_rows (qsa env QWEN38_IDX_ROWWRITE): the writer scatters only the rows the caller reads.
        wide lanes form: row u of lane u's columns (consumed_rows=1, valid_length_lane_rows=rows);
        MTP lanes form : rows u*R .. u*R+R-1 of lane u's columns (consumed_rows=R, valid_length_lane_rows=rows);
        B=1 chain form : row 0 (consumed_rows=1, no lane rows).

Layout (qsa.py): flat cache [1,1,B*rows,128], rows = blocks + 32 keys per lane (lane u's keys at rows u*rows ..),
query tile [1,1,32,128] (Hi=1), gate [1,1,32,1], chunk_start = B*rows - 32 so every key is visible. Output
[1,1,32,B*rows]; row u is read at columns u*rows .. u*rows+blocks.

Test 1 checks, on the consumed rows/columns only, torch.equal against KC=1 / ROWWRITE=0. The output is allocated by the op, so
"sentinel fill" is emulated: a tensor of the output shape is filled with a sentinel and freed right before each run, so
the allocator hands the op the same DRAM back (any row the writer skips keeps the sentinel; the test also asserts that
ROWWRITE=1 leaves a non-consumed row of a written tile unwritten, as a sanity check on the sentinel trick).
Test 2 logs traced us/call.

Run via the go.sh next to the report (flock + timeout 600).
"""

import time

import pytest
import torch
from loguru import logger

import ttnn

pytestmark = pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="indexer_score is Blackhole-only")

DIM = 128
SENTINEL = 12345.0
KCS = [1, 2, 4, 8]
# (lanes B, blocks per lane): total key tiles B * (blocks + 32) / 32 >= 2056 in every case.
SHAPES = [(1, 66048), (4, 16640), (4, 65536), (8, 8192)]


def _inputs(lanes, blocks, seed=7):
    g = torch.Generator().manual_seed(seed)
    rows = blocks + 32
    q = torch.randn(1, 1, 32, DIM, generator=g, dtype=torch.bfloat16)
    k = torch.randn(1, 1, lanes * rows, DIM, generator=g, dtype=torch.bfloat16)
    w = torch.randn(1, 1, 32, 1, generator=g, dtype=torch.bfloat16)
    return q, k, w, rows


def _dev(t, device, **kw):
    return ttnn.from_torch(t, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, **kw)


def _kc_cfg(kc):
    return {} if kc == 1 else {"program_config": ttnn.IndexerScoreProgramConfig(k_chunk_size=32 * kc)}


def _lens(lanes, blocks, device):
    # per-lane valid lengths in key rows (kv_block_start analogue); bound = (len >> 5) + 1 tiles. Mixed, one full.
    vals = [min(blocks, (blocks // 3) + 37 * i + (i % 3) * 1000) for i in range(lanes)]
    vals[-1] = blocks
    t = ttnn.from_torch(
        torch.tensor(vals, dtype=torch.int32).reshape(1, 1, 1, lanes),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return vals, t


def _sentinel_then_free(device, total_cols):
    s = ttnn.from_torch(
        torch.full((1, 1, 32, total_cols), SENTINEL, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.synchronize_device(device)
    ttnn.deallocate(s)


def _run(device, q_d, k_d, w_d, lanes, rows, *, kc=1, consumed=0, lane_rows=0, len_t=None):
    _sentinel_then_free(device, lanes * rows)
    kwargs = dict(chunk_start_idx=lanes * rows - 32, **_kc_cfg(kc))
    if len_t is not None:
        kwargs["valid_length_tensor"] = len_t
    if lane_rows:
        kwargs["valid_length_lane_rows"] = lane_rows
    if consumed:
        kwargs["consumed_rows"] = consumed
    out = ttnn.experimental.indexer_score_dsa(q_d, k_d, w_d, **kwargs)
    res = ttnn.to_torch(out)
    ttnn.deallocate(out)
    return res


def _live_cols(lane, rows, blocks, vals):
    lo = lane * rows
    if vals is None:
        return lo, lo + blocks
    bound = min((vals[lane] >> 5) + 1, rows // 32)
    return lo, lo + min(bound * 32, blocks)


@pytest.mark.parametrize("with_len", [False, True], ids=["nolen", "len"])
@pytest.mark.parametrize("lanes,blocks", SHAPES, ids=[f"B{b}_blk{n}" for b, n in SHAPES])
def test_lanes_kc_rowwrite_bitwise(device, lanes, blocks, with_len):
    assert lanes * (blocks + 32) // 32 >= 2056
    q, k, w, rows = _inputs(lanes, blocks)
    q_d, k_d, w_d = _dev(q, device), _dev(k, device), _dev(w, device)
    vals, len_t = _lens(lanes, blocks, device) if with_len else (None, None)
    lane_rows = rows if with_len else 0

    def consumed_view(res, kc_label):
        # lane u's row u over its live columns
        return [res[0, 0, u, slice(*_live_cols(u, rows, blocks, vals))].clone() for u in range(lanes)]

    base = _run(device, q_d, k_d, w_d, lanes, rows, kc=1, lane_rows=lane_rows, len_t=len_t)
    base_rows = consumed_view(base, "kc1")

    for kc in KCS:
        for row_write in (0, 1):
            if kc == 1 and row_write == 0:
                continue
            res = _run(
                device,
                q_d,
                k_d,
                w_d,
                lanes,
                rows,
                kc=kc,
                consumed=1 if row_write else 0,
                lane_rows=rows if (row_write or with_len) else 0,
                len_t=len_t,
            )
            got = consumed_view(res, f"kc{kc}")
            for u in range(lanes):
                assert torch.equal(got[u], base_rows[u]), f"KC={kc} ROWWRITE={row_write} lane {u} differs from KC=1"
            if row_write:
                # sanity: a non-consumed row of lane 0's first tile was NOT written (still the sentinel or old data
                # of a previous allocation); informational only, the allocator may not reuse the address.
                other = res[0, 0, (1 % 32) if lanes > 1 else 5, 0:32]
                logger.info(
                    f"B={lanes} KC={kc}: non-consumed row sample unwritten={bool((other.float() == SENTINEL).all())}"
                )


@pytest.mark.parametrize("kc", [1, 4])
def test_mtp_lanes_rows_bitwise(device, kc):
    # MTP lanes form: R rows per lane, lane u = rows u*R .. u*R+R-1 (qsa lane_of_row r = r // R)
    lanes, blocks, R = 4, 16640, 4
    q, k, w, rows = _inputs(lanes, blocks, seed=11)
    q_d, k_d, w_d = _dev(q, device), _dev(k, device), _dev(w, device)
    base = _run(device, q_d, k_d, w_d, lanes, rows, kc=1)
    res = _run(device, q_d, k_d, w_d, lanes, rows, kc=kc, consumed=R, lane_rows=rows)
    for u in range(lanes):
        lo = u * rows
        for j in range(R):
            r = u * R + j
            assert torch.equal(res[0, 0, r, lo : lo + blocks], base[0, 0, r, lo : lo + blocks]), f"lane {u} row {r}"


@pytest.mark.parametrize("kc", [1, 4])
def test_chain_b1_row0_bitwise(device, kc):
    # B=1 chain form: single lane, no lane rows, row 0 only
    blocks = 66048
    q, k, w, rows = _inputs(1, blocks, seed=13)
    q_d, k_d, w_d = _dev(q, device), _dev(k, device), _dev(w, device)
    base = _run(device, q_d, k_d, w_d, 1, rows, kc=1)
    res = _run(device, q_d, k_d, w_d, 1, rows, kc=kc, consumed=1)
    assert torch.equal(res[0, 0, 0, :blocks], base[0, 0, 0, :blocks])


TIMING = [(4, 65536), (8, 8192)]


@pytest.mark.parametrize("device_params", [{"trace_region_size": 32 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("row_write", [0, 1], ids=["rw0", "rw1"])
@pytest.mark.parametrize("kc", KCS, ids=[f"kc{k}" for k in KCS])
@pytest.mark.parametrize("lanes,blocks", TIMING, ids=[f"B{b}_blk{n}" for b, n in TIMING])
def test_lanes_kc_rowwrite_traced_us(device, lanes, blocks, kc, row_write):
    q, k, w, rows = _inputs(lanes, blocks)
    q_d, k_d, w_d = _dev(q, device), _dev(k, device), _dev(w, device)
    kwargs = dict(chunk_start_idx=lanes * rows - 32, **_kc_cfg(kc))
    if row_write:
        kwargs.update(consumed_rows=1, valid_length_lane_rows=rows)

    def call():
        o = ttnn.experimental.indexer_score_dsa(q_d, k_d, w_d, **kwargs)
        ttnn.deallocate(o)

    call()  # compile
    ttnn.synchronize_device(device)
    n_per_trace, replays = 10, 5
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(n_per_trace):
        call()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)  # warm
    t0 = time.perf_counter()
    for _ in range(replays):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    us = (time.perf_counter() - t0) / (replays * n_per_trace) * 1e6
    ttnn.release_trace(device, tid)
    logger.info(f"TRACED_US B={lanes} blocks={blocks} KC={kc} ROWWRITE={row_write}: {us:.1f} us/call")
    print(f"TRACED_US B={lanes} blocks={blocks} KC={kc} ROWWRITE={row_write} {us:.1f}")
