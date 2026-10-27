# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The mHC site's TP sum on the decode tile as one all-gather and a sum (``tt/v41/mhc.py``,
``V41_MHC_TP_SUM=gather``)
against ``TtMHCWrap._tp_sum`` (``all_reduce_async``: an all-broadcast and four layout ops around the same
sum).

No weights: a ``_V41Site`` with a random projection, ``V41_TS_TRIALS`` random fp32 ``[1, 1, 32, 32]`` tiles a
chip
(every chip its own), the gathered sum compared bit for bit with the all-reduce's on every chip, then each
path
under trace (``V41_TS_REPLAYS`` replays of {calls} calls) for the per-site device time, and the prefill shape
(``T``
5,120 / sp) checked to keep the all-reduce. Results: ``V41_TS_RESULT``, ``V41_TS_OUT``."""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import TtMHCWrap
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.v41 import mhc as mhc_mod
from models.demos.deepseek_v3_d_p.tt.v41.mhc import _V41Site, mhc_config


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device, device_params", galaxy_meshes(trace_region_size=64 * 1024 * 1024), indirect=True)
def test_v41_mhc_tp_sum(mesh_device, device_params):
    mode = mhc_mod._TP_SUM
    assert mode == "gather", "run with V41_MHC_TP_SUM=gather"
    trials = int(os.environ.get("V41_TS_TRIALS", 4))
    replays = int(os.environ.get("V41_TS_REPLAYS", 20))
    calls = int(os.environ.get("V41_TS_CALLS", 32))  # calls a trace: the replay's fixed cost divided by this
    sp, tp = tuple(mesh_device.shape)
    cfg = mhc_config(C)
    torch.manual_seed(11)
    mix = (2 + cfg.n) * cfg.n
    fn = torch.randn(mix, cfg.n * cfg.dim) * (cfg.n * cfg.dim) ** -0.5
    base, scale = torch.randn(mix) * 0.1, torch.tensor([1.0, 1.0, 1.0])
    site = _V41Site(mesh_device, cfg, fn, base, scale, tp_axis=1, topology=per_axis_topology()[1])
    composite = lambda t: TtMHCWrap._tp_sum(site, t)

    def tiles(rows: int, seed: int):
        gen = torch.Generator().manual_seed(seed)
        t = torch.randn(sp * rows, tp * 32, generator=gen) * 4.0  # [sp * rows, tp * 32]: every chip its own tile
        return ttnn.from_torch(
            t.view(1, 1, sp * rows, tp * 32),
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, (sp, tp), dims=(2, 3)),
        )

    result = {"mesh": [sp, tp], "mode": mode, "trials": []}
    for trial in range(trials):
        t = tiles(32, trial)
        ref, out = composite(t), site._tp_sum(t)
        assert tuple(out.shape) == tuple(ref.shape) == (1, 1, 32, 32), (tuple(out.shape), tuple(ref.shape))
        diffs = []
        for i, (a, b) in enumerate(zip(ttnn.get_device_tensors(ref), ttnn.get_device_tensors(out))):
            a, b = ttnn.to_torch(a), ttnn.to_torch(b)
            n = int((a != b).sum())
            diffs.append(n)
        # the chips of one TP row read the same sum
        rows = [ttnn.to_torch(c)[0, 0] for c in ttnn.get_device_tensors(out)]
        same = all(torch.equal(rows[r * tp], rows[r * tp + c]) for r in range(sp) for c in range(tp))
        a, b = (
            ttnn.to_torch(ttnn.get_device_tensors(ref)[0]).double(),
            ttnn.to_torch(ttnn.get_device_tensors(out)[0]).double(),
        )
        rel = float((a - b).abs().max() / a.abs().max())
        entry = {
            "trial": trial,
            "words_differ": sum(diffs),
            "chips_differ": sum(1 for d in diffs if d),
            "tp_same": same,
            "rel_max": rel,
        }
        result["trials"].append(entry)
        logger.info(f"mhc tp sum trial {json.dumps(entry)}")
        assert same and sum(diffs) == 0, entry  # the composite's own reduction on the same tiles: bit for bit
        for x in (t, ref, out):
            ttnn.deallocate(x)
    # the prefill shape keeps the all-reduce (the gather path is the decode tile's)
    big = tiles(5120 // sp, 99)
    out = site._tp_sum(big)
    assert tuple(out.shape) == (1, 1, 5120 // sp, 32), tuple(out.shape)
    for x in (big, out):
        ttnn.deallocate(x)
    # the traced per-site time of each path
    t = tiles(32, 7)
    timing = {}
    for name, fn_ in (("all_reduce", lambda: composite(t)), (mode, lambda: site._tp_sum(t))):
        for _ in range(2):
            ttnn.deallocate(fn_())
        ttnn.synchronize_device(mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        outs = [fn_() for _ in range(calls)]
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        best = None
        for _ in range(replays):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            dt = (time.perf_counter() - t0) / calls
            best = dt if best is None else min(best, dt)
        ttnn.release_trace(mesh_device, tid)
        for o in outs:
            ttnn.deallocate(o)
        timing[name] = round(best * 1e6, 1)
        logger.info(f"mhc tp sum traced {name}: {timing[name]} us a site (min of {replays} replays of {calls} calls)")
    result["site_us"] = timing
    # the pieces: each gather dim alone, the sum alone on a resident gathered tensor
    ccl = site.tt_ccl
    gather = lambda t, dim: ttnn.experimental.all_gather_async(
        t,
        dim=dim,
        multi_device_global_semaphore=ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=1),
        barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=1),
        num_links=site.num_links,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=site.topology,
        cluster_axis=1,
    )
    g1 = gather(t, 1)
    pieces = {
        "gather_dim1": lambda: gather(t, 1),
        "gather_dim2": lambda: gather(t, 2),
        "gather_dim3": lambda: gather(t, 3),
        "sum_dim1": lambda: ttnn.sum(g1, dim=1, keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        "sum_dim1_nokeep": lambda: ttnn.sum(g1, dim=1, keepdim=False, memory_config=ttnn.DRAM_MEMORY_CONFIG),
    }
    piece_us = {}
    for name, fn_ in pieces.items():
        for _ in range(2):
            ttnn.deallocate(fn_())
        ttnn.synchronize_device(mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        outs = [fn_() for _ in range(calls)]
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        best = None
        for _ in range(replays):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            dt = (time.perf_counter() - t0) / calls
            best = dt if best is None else min(best, dt)
        ttnn.release_trace(mesh_device, tid)
        for o in outs:
            ttnn.deallocate(o)
        piece_us[name] = round(best * 1e6, 1)
        logger.info(f"mhc tp sum piece {name}: {piece_us[name]} us (min of {replays} replays of {calls} calls)")
    result["piece_us"] = piece_us
    logger.info(f"V41_TS_RESULT {json.dumps(result)}")
    out_path = os.environ.get("V41_TS_OUT")
    if out_path:
        with open(os.path.expanduser(out_path), "w") as f:
            json.dump(result, f, indent=1)
