# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Traced latency of the sparse-MLA piece on one chip, at the 1M-context geometry (32768 rows of every user on the
chip): each program captured as a trace of its own and replayed 20 times back to back with one synchronize at the
end, the method of tests/test_perf_decode.py. Next to it, the ttnn-op form of the same stage as PERF.md 3.2
measured it at 8 users x 1M (per user: row gather 1.207 ms + collect 0.184 ms + attention core 0.244 ms).

Also the row read rate of the reader alone, over the number of reader cores.

Mock Galaxy (HY4_MOCK=1): the programs are compiled, nothing is timed."""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.sparse_mla import layout as lo
from models.experimental.hy4_preview.fused.sparse_mla.op import SparseMla
from models.experimental.hy4_preview.fused.sparse_mla.tests import harness as hs
from models.experimental.hy4_preview.fused.sparse_mla.tests.test_attend import check_partials

REPLAYS = 20
TTNN_FORM_MS_PER_USER = {"row gather": 1.207, "collect": 0.184, "attention core": 0.244}  # PERF.md 3.2, 8 users x 1M
ROWS_PER_SLOT = 32768  # one user's rows on a chip at 1M context

trace_device = pytest.mark.parametrize("device_params", [{"trace_region_size": 4 << 20}], indirect=True)


def _sync(device) -> float:
    ttnn.synchronize_device(device)
    return time.perf_counter()


def replay_us(device, fn) -> float:
    """us per replay of ``fn`` captured as a trace: REPLAYS replays enqueued back to back, one synchronize."""
    fn()  # compile and cache the program outside the capture
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    fn()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    start = _sync(device)
    for _ in range(REPLAYS):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    us = 1e6 * (_sync(device) - start) / REPLAYS
    ttnn.release_trace(device, trace_id)
    return us


def _storage(device, cfg):
    latent, rope = hs.random_latents(cfg.users, cfg.rows_per_slot, cfg, seed=3)
    return hs.make_storage(device, cfg, cfg.users, latent, rope)


@trace_device
@pytest.mark.parametrize("fidelity", ["lofi"])
def test_attend_latency(device, fidelity):
    from models.experimental.hy4_preview.fused.sparse_mla.tests.test_attend import FIDELITY

    cfg = lo.SparseMlaConfig(rows_per_slot=ROWS_PER_SLOT, num_layers=1)
    op = SparseMla(cfg, device)
    kv, rows = _storage(device, cfg)
    gen = torch.Generator().manual_seed(7)
    q_lat = torch.randn(cfg.users, cfg.heads, cfg.width, generator=gen)
    sink = torch.randn(cfg.heads, generator=gen)
    q = hs.to_dev(op.queries_to_pages(q_lat), device, ttnn.bfloat16)
    ttnn_ms = sum(TTNN_FORM_MS_PER_USER.values())

    results = []
    for users, n in [(1, 64), (1, 128), (1, 2048), (8, 64), (8, 128), (8, 512), (8, 2048)]:
        lists = [hs.spread_ids(cfg, n if u < users else 0, seed=30 + u) for u in range(cfg.users)]
        sel = hs.make_selection(device, cfg, lists)
        out = hs.to_dev(torch.zeros(1, 1, cfg.users * cfg.head_groups, op.out_page_values), device, ttnn.bfloat16)
        program = op.attend_program(kv, sel, q, out, 0, fidelity=FIDELITY[fidelity])
        if hs.MOCK:
            ttnn.experimental.prepare_generic_op([kv, sel, q, out], program)
            continue
        us = replay_us(device, lambda: ttnn.generic_op([kv, sel, q, out], program))
        got = op.pages_to_partials(ttnn.to_torch(out))
        check_partials(cfg, got, rows, lists, q_lat, sink, 0, f"latency {users}x{n}", floor=0.99)
        results.append((users, n, us))
        logger.info(
            f"HY4_MK3 latency attend {fidelity}: {users} query rows x {n} rows per chip: {us:8.1f} us per program "
            f"(traced, {REPLAYS} replays) | ttnn form {users} x {ttnn_ms:.3f} ms = {users * ttnn_ms * 1e3:.0f} us "
            f"| x{users * ttnn_ms * 1e3 / us:.0f}"
        )
        for t in (sel, out):
            ttnn.deallocate(t)


@trace_device
def test_read_rate(device):
    cfg = lo.SparseMlaConfig(rows_per_slot=ROWS_PER_SLOT, num_layers=1)
    op = SparseMla(cfg, device)
    kv, _ = _storage(device, cfg)
    lists = [hs.spread_ids(cfg, cfg.topk, seed=30 + u) for u in range(cfg.users)]
    sel = hs.make_selection(device, cfg, lists)
    empty = hs.make_selection(device, cfg, [hs.spread_ids(cfg, 0, seed=0)] * cfg.users)
    total = cfg.users * cfg.topk
    # readers in the two columns next to the DRAM banks first, then the columns beside them
    order = [(x, y) for x in (0, 7, 1, 8, 2, 9) for y in range(8)]
    for n in (1, 2, 4, 8, 16, 24, 32, 48):
        cores = sorted(order[:n], key=lambda c: (c[1], c[0])) if n <= 16 else order[:n]
        program = op.read_only_program(kv, sel, 0, cores)
        idle = op.read_only_program(kv, empty, 0, cores)
        if hs.MOCK:
            ttnn.experimental.prepare_generic_op([kv, sel], program)
            continue
        us = replay_us(device, lambda: ttnn.generic_op([kv, sel], program))
        floor = replay_us(device, lambda: ttnn.generic_op([kv, empty], idle))
        rate = total / ((us - floor) * 1e-6)
        logger.info(
            f"HY4_MK3 read rate {n:2d} reader cores: {total} rows (8 users x {cfg.topk}, spread over the slab) in "
            f"{us:8.1f} us, program with empty lists {floor:6.1f} us -> {rate / 1e6:6.2f} M rows/s = "
            f"{rate * cfg.row_stride / 1e9:6.2f} GB/s of DRAM pages ({rate * cfg.row_bytes / 1e9:6.2f} GB/s of row bytes)"
        )


@trace_device
def test_append_latency(device):
    """The single-row append of 8 users (all 8 positions owned by this chip: the worst case for one chip)."""
    cfg = lo.SparseMlaConfig(rows_per_slot=ROWS_PER_SLOT, num_layers=1)
    op = SparseMla(cfg, device)
    kv, _ = _storage(device, cfg)
    chip = 9
    gen = torch.Generator().manual_seed(2)
    rows = hs.pack_rows(
        device,
        torch.randn(cfg.users, 1, 1, cfg.latent, generator=gen),
        torch.randn(cfg.users, 1, 1, cfg.rope, generator=gen),
    )
    positions = [(100 + 3 * u) * 4096 + chip * 128 + 5 * u for u in range(cfg.users)]
    pos = hs.to_dev(torch.tensor(positions, dtype=torch.int32).reshape(1, 1, 1, -1), device, ttnn.uint32)
    program = op.append_program(kv, rows, pos, 0, chip)
    if hs.MOCK:
        ttnn.experimental.prepare_generic_op([rows, pos, kv], program)
        return
    us = replay_us(device, lambda: ttnn.generic_op([rows, pos, kv], program))
    logger.info(f"HY4_MK3 latency append: 8 packed rows written by one chip: {us:6.1f} us per program (traced)")


@trace_device
def test_stage_split(device):
    """Where the time of the attention program goes: the readers alone (16 cores), readers + converters (K tiles
    produced and dropped), the attention program with kernels that return at once (what its launch on 80 cores
    costs), and the whole program, on the same lists."""
    from models.experimental.hy4_preview.fused.sparse_mla.tests.test_attend import FIDELITY

    cfg = lo.SparseMlaConfig(rows_per_slot=ROWS_PER_SLOT, num_layers=1)
    op = SparseMla(cfg, device)
    kv, _ = _storage(device, cfg)
    gen = torch.Generator().manual_seed(7)
    q = hs.to_dev(
        op.queries_to_pages(torch.randn(cfg.users, cfg.heads, cfg.width, generator=gen)), device, ttnn.bfloat16
    )
    dump = hs.to_dev(torch.zeros(1, 1, cfg.users * cfg.topk, cfg.width), device, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    readers = [c for blk in op.blocks for c in blk.readers]
    for users, n in [(8, 0), (1, 64), (8, 64), (8, 128), (8, 2048)]:
        lists = [hs.spread_ids(cfg, n if u < users else 0, seed=30 + u) for u in range(cfg.users)]
        sel = hs.make_selection(device, cfg, lists)
        out = hs.to_dev(torch.zeros(1, 1, cfg.users * cfg.head_groups, op.out_page_values), device, ttnn.bfloat16)
        programs = {
            "read": ([kv, sel], op.read_only_program(kv, sel, 0, readers)),
            "read+convert": ([kv, sel, dump], op.read_rows_program(kv, sel, dump, 0, drop=True)),
            "launch (kernels return at once)": ([kv, sel, q, out], op.attend_program(kv, sel, q, out, 0, nop=True)),
            "attend": ([kv, sel, q, out], op.attend_program(kv, sel, q, out, 0, fidelity=FIDELITY["lofi"])),
        }
        line = []
        for name, (tensors, program) in programs.items():
            if hs.MOCK:
                ttnn.experimental.prepare_generic_op(tensors, program)
                continue
            line.append(f"{name} {replay_us(device, lambda: ttnn.generic_op(tensors, program)):8.1f} us")
        logger.info(f"HY4_MK3 stage split {users} query rows x {n} rows per chip: " + " | ".join(line))
        for t in (sel, out):
            ttnn.deallocate(t)


@trace_device
def test_launch_shapes(device):
    """What a program costs before any kernel does anything, over its shape: kernels that return at once
    (kernels/sm_nop.cpp) on the attention cores (64 cores, two 4x8 rectangles) and on all 80 cores of the piece."""
    cfg = lo.SparseMlaConfig(rows_per_slot=ROWS_PER_SLOT, num_layers=1)
    op = SparseMla(cfg, device)
    anchor = [hs.to_dev(torch.zeros(1, 1, 1, 32), device, ttnn.bfloat16) for _ in range(2)]  # generic_op wants two
    one = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))])
    row = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(4, 0))])
    shapes = [
        ("1 core, 1 DM kernel", one, dict(dm=1)),
        ("1 core, 2 DM + compute", one, dict(dm=2, compute=True)),
        ("4 cores, 2 DM + compute", row, dict(dm=2, compute=True)),
        ("64 cores, 1 DM kernel", op.attention, dict(dm=1)),
        ("64 cores, 2 DM kernels", op.attention, dict(dm=2)),
        ("64 cores, compute only", op.attention, dict(dm=0, compute=True)),
        ("64 cores, 2 DM + compute", op.attention, dict(dm=2, compute=True)),
        ("64 cores, 2 DM + compute, 16 CBs", op.attention, dict(dm=2, compute=True, cbs=16)),
        (
            "64 cores, 2 DM + compute, 16 CBs, 6 semaphores",
            op.attention,
            dict(dm=2, compute=True, cbs=16, semaphores=6),
        ),
        ("80 cores, 2 DM + compute, 16 CBs, 6 semaphores", op.both, dict(dm=2, compute=True, cbs=16, semaphores=6)),
    ]
    for name, cores, kwargs in shapes:
        program = op.nop_program(cores, **kwargs)
        if hs.MOCK:
            ttnn.experimental.prepare_generic_op(anchor, program)
            continue
        us = replay_us(device, lambda: ttnn.generic_op(anchor, program))
        logger.info(f"HY4_MK3 launch shape {name}: {us:6.1f} us per program (traced, kernels return at once)")
