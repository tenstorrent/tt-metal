# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device test + microbenchmark for QWEN36_DECODE_AR_LIVE_ROWS (all_reduce_async live_rows_packing).

The decode all-reduce (DecodeResidualAllReduce, RES40: [1,1,R,5120] bf16 width-sharded over 40 cores, persistent
[32, 5120*tp] buffer) always sends full 32-row tiles. With live_rows_packing the op sends only rows 0..15 of each tile
(faces 0/1), four half-tiles per 4 KB fabric packet, when the partial has <= 16 logical rows.

test_decode_ar_live_rows_correctness: through the model's own DecodeResidualAllReduce.all_reduce, rows 0..R-1 of the
packed result are byte-identical to the default result and match the torch sum, for R in (1, 4, 8, 16, 32); R = 32
(ineligible) is byte-identical over all 32 rows.

test_decode_ar_live_rows_bench: traced latency of the bare all_reduce_async (same RES40 configs, links, topology as the
model) at R = 4, 8, 16, 32 with packing off and on. Logs one line per case plus the two numbers the plan's gate needs:
off(R=8) - off(R=32) (should be ~0: the default always sends 32 rows) and off(R=8) - on(R=8) (the per-AR saving). The
port is worth enabling only if that saving is >= 10 us per AR. QWEN36_AR_BENCH_MIN_GAIN_US=<us> turns the gate into
an assertion (default: log only).

Run (Galaxy column, ring fabric, as served):
    QWEN36_FABRIC_RING=1 MESH_DEVICE=P150x8 \
        pytest models/demos/blackhole/qwen36/tests/test_decode_ar_live_rows.py -v -s
"""

import os
import time
import types

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc

DIM = 5120  # Qwen3.8-27B hidden size (RES40 shard width 128)


def _parametrize_traced(max_tp=8, trace_bytes=64 * 1024 * 1024):
    from models.demos.blackhole.qwen36.tests.test_factory import _resolve_mesh_shape

    shape = _resolve_mesh_shape(max_tp)

    def decorator(fn):
        fn = pytest.mark.parametrize(
            "device_params",
            [
                {
                    # QWEN36_FABRIC_RING=1: a Galaxy column's Ring CCLs need the ring (torus) fabric.
                    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING
                    if os.environ.get("QWEN36_FABRIC_RING") == "1"
                    else ttnn.FabricConfig.FABRIC_1D,
                    "trace_region_size": trace_bytes,
                }
            ],
            indirect=True,
        )(fn)
        fn = pytest.mark.parametrize("mesh_device", [pytest.param(shape, id=f"{shape[0]}x{shape[1]}")], indirect=True)(
            fn
        )
        return fn

    return decorator


def _topology():
    return ttnn.Topology.Ring if os.environ.get("QWEN36_FABRIC_RING") == "1" else ttnn.Topology.Linear


def _dar(mesh_device, live_rows):
    from models.tt_transformers.tt.ccl import TT_CCL

    args = types.SimpleNamespace(dim=DIM, ccl_topology=_topology)
    return tpc.DecodeResidualAllReduce(mesh_device, args, TT_CCL(mesh_device), live_rows=live_rows)


def _partials(mesh_device, rows, seed):
    """Per-device distinct partials [1,1,rows,DIM] (device j gets slice j of a [1,tp,rows,DIM] tensor) + their sum."""
    tp = list(mesh_device.shape)[1]
    g = torch.Generator().manual_seed(seed)
    full = torch.randn(1, tp, rows, DIM, generator=g).to(torch.bfloat16)
    return full, full.float().sum(dim=1, keepdim=True)


def _to_mesh(mesh_device, full, memory_config):
    return ttnn.from_torch(
        full,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
    )


def _per_device(t):
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


@torch.no_grad()
@_parametrize_traced()
def test_decode_ar_live_rows_correctness(mesh_device, reset_seeds, ensure_gc):
    if list(mesh_device.shape)[1] < 2:
        pytest.skip("needs a (1, N>=2) mesh")
    dar_off = _dar(mesh_device, live_rows=False)
    dar_on = _dar(mesh_device, live_rows=True)
    assert dar_on.live_rows and not dar_off.live_rows

    for rows in (1, 4, 8, 16, 32):
        full, ref = _partials(mesh_device, rows, seed=rows)
        outs = {}
        for name, dar in (("off", dar_off), ("on", dar_on)):
            # all_reduce reshards + deallocates its input: a fresh DRAM partial per call.
            partial = _to_mesh(mesh_device, full, ttnn.DRAM_MEMORY_CONFIG)
            out = dar.all_reduce(partial)
            ttnn.synchronize_device(mesh_device)
            outs[name] = _per_device(out)
            ttnn.deallocate(out)
        for dev, (o_off, o_on) in enumerate(zip(outs["off"], outs["on"])):
            assert o_on.shape[-2] == rows, f"rows={rows}: output logical rows {o_on.shape[-2]}"
            live_off, live_on = o_off[..., :rows, :], o_on[..., :rows, :]
            assert torch.equal(live_off, live_on), (
                f"rows={rows} device {dev}: live rows differ between packing off and on "
                f"(max|d|={(live_off.float() - live_on.float()).abs().max().item():.3e})"
            )
            err = (live_on.float() - ref).abs().max().item()
            tol = 0.05 * ref.abs().max().item()  # bf16 partials, bf16 dest: a few ulps of the largest sum
            assert err <= tol, f"rows={rows} device {dev}: |all_reduce - torch sum| = {err:.3e} > {tol:.3e}"
        logger.info(f"rows={rows}: live rows byte-identical (packing off vs on) on {len(outs['on'])} devices")


def _bench_one(mesh_device, dar, rows, live_rows_packing, n_iters, n_buffers=4):
    """Traced per-AR latency (us) of n_iters back-to-back all_reduce_async calls on a pre-packed RES40 partial.

    Rotates n_buffers persistent buffers + global semaphores (as tests/ttnn/.../test_new_all_reduce.py does) so
    consecutive calls never share a staging buffer; the buffers are zero-filled like the live-rows model buffer."""
    tp = dar.tp
    full, _ = _partials(mesh_device, rows, seed=1000 + rows)
    packed = _to_mesh(mesh_device, full, dar.res_mem)
    grid = mesh_device.compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sems = [ttnn.create_global_semaphore(mesh_device, all_cores, 0) for _ in range(n_buffers)]
    buf_mem = dar.buffer.memory_config()
    bufs = [
        ttnn.from_torch(
            torch.zeros([1, 1, ttnn.TILE_SIZE, DIM * tp]),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=buf_mem,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        for _ in range(n_buffers)
    ]
    extra = dict(live_rows_packing=True) if live_rows_packing else {}

    def run(n):
        out = None
        for i in range(n):
            if out is not None:
                ttnn.deallocate(out)
            out = ttnn.experimental.all_reduce_async(
                packed,
                bufs[i % n_buffers],
                cluster_axis=1,
                mesh_device=mesh_device,
                multi_device_global_semaphore=sems[i % n_buffers],
                topology=dar.topology,
                num_links=dar.num_links,
                memory_config=dar.res_mem,
                **extra,
            )
        return out

    ttnn.deallocate(run(n_buffers))  # compile
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = run(n_iters)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)  # warm replay
    ttnn.synchronize_device(mesh_device)
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        best = min(best, time.perf_counter() - t0)
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.deallocate(out)
    for t in bufs + [packed]:
        ttnn.deallocate(t)
    return best / n_iters * 1e6


@torch.no_grad()
@_parametrize_traced()
def test_decode_ar_live_rows_bench(mesh_device, reset_seeds, ensure_gc):
    if list(mesh_device.shape)[1] < 2:
        pytest.skip("needs a (1, N>=2) mesh")
    n_iters = int(os.environ.get("QWEN36_AR_BENCH_ITERS", "200"))
    dar = _dar(mesh_device, live_rows=False)
    us = {}
    for rows in (4, 8, 16, 32):
        for packing in (False, True):
            us[(rows, packing)] = _bench_one(mesh_device, dar, rows, packing, n_iters)
            logger.info(
                f"AR_BENCH rows={rows:2d} live_rows_packing={int(packing)}: {us[(rows, packing)]:.2f} us/AR "
                f"(num_links={dar.num_links}, topology={dar.topology}, iters={n_iters})"
            )
    rows_cost = us[(8, False)] - us[(32, False)]
    gain8 = us[(8, False)] - us[(8, True)]
    gain16 = us[(16, False)] - us[(16, True)]
    logger.info(
        f"AR_BENCH summary: off(8)-off(32) = {rows_cost:+.2f} us (default path ignores live rows); "
        f"saving at 8 rows = {gain8:+.2f} us/AR, at 16 rows = {gain16:+.2f} us/AR; "
        f"32 rows on-vs-off = {us[(32, True)] - us[(32, False)]:+.2f} us (must be ~0: fallback). "
        f"Plan gate: enable only if the 8-row saving is >= 10 us/AR (~146 ARs per spec step)."
    )
    min_gain = os.environ.get("QWEN36_AR_BENCH_MIN_GAIN_US")
    if min_gain is not None:
        assert gain8 >= float(min_gain), f"8-row live-rows saving {gain8:.2f} us/AR < {min_gain} us"
