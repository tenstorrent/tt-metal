# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Rebuild stage 1: ternary decode linears fed by ttnn.dram_prefetcher through a GlobalCircularBuffer ring."""

import time

import numpy as np
import pytest
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.prefetch.consumer import GcbTernaryLinear, balanced
from models.demos.blackhole.bonsai2.tt.prefetch.stream import (
    NUM_RECEIVERS,
    Prefetch,
    WeightStream,
    page_tiles,
    tiles_to_pages,
)
from models.demos.blackhole.bonsai2.tt.ternary.decode_linear3 import TernaryDecodeLinear3, reference

PARAMS = [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 60_000_000}]


def test_page_tiles_roundtrip():
    p = np.random.default_rng(0).integers(0, 2**32, size=(5, 1024), dtype=np.uint32)
    assert (tiles_to_pages(page_tiles(p)) == p).all()


def _weights(L, K, N, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(L):
        t = rng.integers(-1, 2, size=(K, N)).astype(np.int8)
        s = (rng.random((N, K // 128)) * 0.02 + 0.005).astype(np.float16)
        out.append((t, s))
    return out


def _build(device, ws_list, N):
    ws = WeightStream()
    lins = []
    for i, (t, s) in enumerate(ws_list):
        lin = GcbTernaryLinear.from_host(t, s, balanced(N // 32, offset=32 * i))
        lin.bind(ws.add(lin.pages()))
        lins.append(lin)
    chunks, tails = ws.build(device)
    assert len(set(tails)) == 1, tails
    return lins, chunks, tails[0]


@pytest.mark.parametrize("device_params", PARAMS, indirect=True)
@pytest.mark.parametrize("L, K, N", [(4, 5120, 8192), (4, 5120, 17408)])
def test_gcb_tdec(device, L, K, N):
    wl = _weights(L, K, N)
    x = torch.randn(K).bfloat16()
    pf = Prefetch(device)
    try:
        lins, chunks, tail = _build(device, wl, N)
        # the chunk tensor holds receiver g's pages in tile column g
        back = ttnn.to_torch(chunks[0]).view(torch.int16).numpy().reshape(-1, NUM_RECEIVERS * 32)
        for g in (0, 9, 63):
            want = lins[0].pages()[g]
            got = tiles_to_pages(back[:, g * 32 : (g + 1) * 32])[: want.shape[0]]
            assert (got == want).all(), g
        act = ttnn.from_torch(
            x.reshape(1, -1).repeat(8, 1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        pf.launch(chunks)
        ys = []
        for lin in lins:
            ys.append(lin(act, pf))
            ttnn.synchronize_device(device)
        pf.drain(tail)
        ttnn.synchronize_device(device)
        for i, ((t, s), y) in enumerate(zip(wl, ys)):
            ref = reference(x, t, s)
            got = ttnn.to_torch(y).float().reshape(-1)[:N]
            rel = ((got - ref).norm() / ref.norm()).item()
            print(f"gcb op {i} K={K} N={N} rel={rel:.5f}")
            assert rel < 0.02, (i, rel)
    finally:
        pf.close()


@pytest.mark.parametrize("device_params", PARAMS, indirect=True)
@pytest.mark.parametrize("L, K, N, R", [(4, 5120, 17408, 20), (8, 17408, 5120, 10)])
def test_gcb_tdec_time(device, L, K, N, R):
    """Traced wall time of R rounds of L ops: prefetcher-fed vs TernaryDecodeLinear3 reading DRAM itself."""
    wl = _weights(L, K, N)
    x = torch.randn(K).bfloat16()
    act = ttnn.from_torch(
        x.reshape(1, -1).repeat(8, 1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    nbytes = None

    pf = Prefetch(device)
    try:
        lins, chunks, tail = _build(device, wl, N)
        nbytes = sum(l.rec.nbytes for l in lins)
        pf.launch(chunks, layers=R + 1)
        for lin in lins:  # warm-up round (compiles, primes the cache)
            lin(act, pf)
        pf.drain(tail)
        ttnn.synchronize_device(device)
        tr = ttnn.begin_trace_capture(device, cq_id=0)
        for _ in range(R):
            for lin in lins:
                lin(act, pf)
            pf.drain(tail)
        ttnn.end_trace_capture(device, tr, cq_id=0)
        t0 = time.perf_counter()
        ttnn.execute_trace(device, tr, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        gcb_s = (time.perf_counter() - t0) / R
        ttnn.release_trace(device, tr)
    finally:
        pf.close()

    base = [TernaryDecodeLinear3.from_host(device, t, s, inline_scales=True) for t, s in wl]
    for b in base:
        b(act)
    ttnn.synchronize_device(device)
    tr = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(R):
        for b in base:
            b(act)
    ttnn.end_trace_capture(device, tr, cq_id=0)
    t0 = time.perf_counter()
    ttnn.execute_trace(device, tr, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    base_s = (time.perf_counter() - t0) / R
    ttnn.release_trace(device, tr)
    print(
        f"GCBTIME L={L} K={K} N={N}: prefetched {gcb_s * 1e6:.1f} us/round ({nbytes / gcb_s / 1e9:.0f} GB/s), "
        f"DRAM-read {base_s * 1e6:.1f} us/round ({nbytes / base_s / 1e9:.0f} GB/s), {nbytes / 1e6:.1f} MB/round"
    )


@pytest.mark.parametrize("device_params", PARAMS, indirect=True)
def test_gcb_discard(device):
    """Bisect: the same chunks drained by the stock discard consumer (no Bonsai kernels)."""
    from models.demos.blackhole.bonsai2.tt.prefetch.stream import BLOCK_PAGES, CHUNK_PAGES

    wl = _weights(4, 5120, 8192)
    pf = Prefetch(device)
    try:
        lins, chunks, tail = _build(device, wl, 8192)
        pf.launch(chunks)
        pf.drain(len(chunks) * CHUNK_PAGES // BLOCK_PAGES)
        ttnn.synchronize_device(device)
    finally:
        pf.close()


@pytest.mark.parametrize("device_params", PARAMS, indirect=True)
@pytest.mark.parametrize("sub", [False, True])
def test_gcb_dram_feed(device, sub):
    """Bisect: the consumer program with its pages read from DRAM (no prefetcher running); sub: with the prefetcher's
    sub-devices and ring loaded."""
    import types

    K, N = 5120, 8192
    ((t, s),) = _weights(1, K, N)
    x = torch.randn(K).bfloat16()
    if sub:
        pf = Prefetch(device)
    else:
        from models.tt_transformers.tt.prefetcher import generate_sender_receiver_mapping

        cores = []
        for _, rs in sorted(
            generate_sender_receiver_mapping(8).items(),
            key=lambda kv: (kv[0][0], [9, 1, 7, 3, 0, 2, 6, 4].index(kv[0][1])),
        ):
            cores += ttnn.corerange_to_cores(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in rs]), row_wise=True
            )
        pf = types.SimpleNamespace(
            receivers=cores,
            receiver_crs=ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores]),
            gcb=None,
            close=lambda: None,
        )
    try:
        lin = GcbTernaryLinear.from_host(t, s, balanced(N // 32))
        lin.bind([(n * lin.ppr, -(-n * lin.ppr // 2)) for _, n in lin.assign])
        w = ttnn.from_torch(
            torch.from_numpy(lin.rec.view(np.int32)),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        act = ttnn.from_torch(
            x.reshape(1, -1).repeat(8, 1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        y = lin(act, pf, dram=w)
        ttnn.synchronize_device(device)
        got = ttnn.to_torch(y).float().reshape(-1)[:N]
        ref = reference(x, t, s)
        rel = ((got - ref).norm() / ref.norm()).item()
        print(f"dram-feed rel={rel:.5f}")
        assert rel < 0.02
    finally:
        pf.close()


@pytest.mark.parametrize("device_params", PARAMS, indirect=True)
@pytest.mark.parametrize("cores", [130, 64])
def test_stock_same_params(device, cores):
    """Bisect: the stock DRAM-reading op on the same weights and device params."""
    K, N = 5120, 8192
    ((t, s),) = _weights(1, K, N)
    x = torch.randn(K).bfloat16()
    lin = TernaryDecodeLinear3.from_host(device, t, s, inline_scales=True, cores=cores)
    act = ttnn.from_torch(
        x.reshape(1, -1).repeat(8, 1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    y = lin(act)
    ttnn.synchronize_device(device)
    got = ttnn.to_torch(y).float().reshape(-1, N)[0]
    rel = ((got - reference(x, t, s)).norm() / reference(x, t, s).norm()).item()
    print(f"stock cores={cores} rel={rel:.5f}")
    assert rel < 0.02


def _spin(crs, n):
    program = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source="models/demos/blackhole/bonsai2/tt/prefetch/kernels/spin.cpp",
                core_ranges=crs,
                compile_time_args=[],
                common_runtime_args=[n],
                config=ttnn.ReaderConfigDescriptor(),
            )
        ],
        semaphores=[],
        cbs=[],
    )
    return program


def _timed(device, fn, R):
    fn()
    ttnn.synchronize_device(device)
    tr = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(R):
        fn()
    ttnn.end_trace_capture(device, tr, cq_id=0)
    t0 = time.perf_counter()
    ttnn.execute_trace(device, tr, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    dt = (time.perf_counter() - t0) / R
    ttnn.release_trace(device, tr)
    return dt


@pytest.mark.parametrize("device_params", PARAMS, indirect=True)
@pytest.mark.parametrize("spin", [0, 4000, 12000])
def test_gcb_overlap(device, spin):
    """L ternary ops each followed by a spin op (a stand-in for the non-weight ops between weight ops): does the
    prefetcher keep streaming during the spin? Prints us/round prefetched vs DRAM-reading, and the spins alone."""
    L, K, N, R = 8, 5120, 8192, 10
    wl = _weights(L, K, N)
    x = torch.randn(K).bfloat16()
    act = ttnn.from_torch(
        x.reshape(1, -1).repeat(8, 1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    dummy = ttnn.from_torch(torch.zeros(1, 32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    pf = Prefetch(device)
    try:
        lins, chunks, tail = _build(device, wl, N)
        sp = _spin(pf.receiver_crs, spin)
        pf.launch(chunks, layers=R + 1)  # _timed runs the round 1 + R times

        def rnd():
            for lin in lins:
                lin(act, pf)
                if spin:
                    ttnn.generic_op([dummy], sp)
            pf.drain(tail)

        t_pf = _timed(device, rnd, R)
        t_spin = _timed(device, lambda: [ttnn.generic_op([dummy], sp) for _ in range(L)], R) if spin else 0.0
    finally:
        pf.close()
    grid = device.compute_with_storage_grid_size()
    allc = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    sp_all = _spin(allc, spin)
    base = [TernaryDecodeLinear3.from_host(device, t, s, inline_scales=True) for t, s in wl]

    def rnd_base():
        for b in base:
            b(act)
            if spin:
                ttnn.generic_op([dummy], sp_all)

    t_base = _timed(device, rnd_base, R)
    print(
        f"OVERLAP spin={spin}: spins alone {t_spin * 1e6:.1f} us/round; prefetched+spin {t_pf * 1e6:.1f}, "
        f"DRAM-read+spin {t_base * 1e6:.1f} us/round"
    )
