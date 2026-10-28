# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The mHC site's projection on the decode tile with two ops fewer (``tt/v41/mhc.py``, ``V41_MHC_PROJ_W=dram``: the
weight read from DRAM by the K-split cores, no L1 copy; ``V41_MHC_RSQRT=fused``: the rsqrt of the mean of squares
applied to the mix's coefficient tile inside the stream mix, no ``ttnn.rsqrt``) against the
served path.

No weights: a ``_V41Site`` with a random projection, ``V41_MD_TRIALS`` random fp32 ``[1, 1, 32, 4 * 1280]`` stream
tiles a chip (every chip its own), ``fused_rms_project`` with each new choice alone and both compared bit for bit
with the served path's on every chip, the prefill shape (``T`` 5,120 / sp) checked to keep the L1 copy, then each
path under trace (``V41_MD_REPLAYS`` replays of ``V41_MD_CALLS`` calls) for the per-site device time. Results:
``V41_MD_RESULT``, ``V41_MD_OUT``."""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.v41.mhc import _V41Site, fused_rms_project, mhc_config


def _trace_us(mesh_device, fn, calls: int, replays: int) -> float:
    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    outs = [fn() for _ in range(calls)]
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
    return round(best * 1e6, 1)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device, device_params", galaxy_meshes(trace_region_size=64 * 1024 * 1024), indirect=True)
def test_v41_mhc_decode_ops(mesh_device, device_params):
    trials = int(os.environ.get("V41_MD_TRIALS", 4))
    replays = int(os.environ.get("V41_MD_REPLAYS", 20))
    calls = int(os.environ.get("V41_MD_CALLS", 32))
    sp, tp = tuple(mesh_device.shape)
    cfg = mhc_config(C)
    torch.manual_seed(119)
    mix = (2 + cfg.n) * cfg.n
    fn = torch.randn(mix, cfg.n * cfg.dim) * (cfg.n * cfg.dim) ** -0.5
    base, scale = torch.randn(mix) * 0.1, torch.tensor([1.0, 1.0, 1.0])
    site = _V41Site(mesh_device, cfg, fn, base, scale, tp_axis=1, topology=per_axis_topology()[1])
    tp_sum = site._tp_sum if site.tp_factor > 1 else None
    local = cfg.n * cfg.dim // tp

    def project(x, weight_in_l1: bool, fused_rsqrt: bool):
        return fused_rms_project(
            x, site.fn_T, site.mix_hc, site.norm_eps, site.tp_factor, tp_sum, weight_in_l1, fused_rsqrt
        )

    def streams(rows: int, seed: int):
        gen = torch.Generator().manual_seed(seed)
        t = torch.randn(sp * rows, tp * local, generator=gen) * 2.0  # every chip its own hidden slice of every stream
        return ttnn.from_torch(
            t.view(1, 1, sp * rows, tp * local),
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, (sp, tp), dims=(2, 3)),
        )

    variants = {"dram": (False, False), "rsqrt": (True, True), "both": (False, True)}
    result = {"mesh": [sp, tp], "trials": []}
    for trial in range(trials):
        x = streams(32, trial)
        ref = project(x, True, False)
        for name, (w_l1, rs) in variants.items():
            out = project(x, w_l1, rs)
            assert tuple(out.shape) == tuple(ref.shape) == (1, 1, 32, 32), (tuple(out.shape), tuple(ref.shape))
            diffs = []
            for a, b in zip(ttnn.get_device_tensors(ref), ttnn.get_device_tensors(out)):
                a, b = ttnn.to_torch(a), ttnn.to_torch(b)
                diffs.append(int((a[..., : site.mix_hc] != b[..., : site.mix_hc]).sum()))  # the Sinkhorn's columns
            entry = {"trial": trial, "variant": name, "words_differ": sum(diffs), "chips_differ": sum(map(bool, diffs))}
            result["trials"].append(entry)
            logger.info(f"mhc decode ops trial {json.dumps(entry)}")
            assert sum(diffs) == 0, entry
            ttnn.deallocate(out)
        for t in (x, ref):
            ttnn.deallocate(t)
    # the prefill shape keeps the L1 copy (one split: every core reads the whole weight) whatever the flag says
    big = streams(5120 // sp, 99)
    out = project(big, False, True)
    assert tuple(out.shape) == (1, 1, 5120 // sp, 32), tuple(out.shape)
    for t in (big, out):
        ttnn.deallocate(t)
    # the traced per-site time of each path
    x = streams(32, 7)
    timing = {"served": _trace_us(mesh_device, lambda: project(x, True, False), calls, replays)}
    for name, (w_l1, rs) in variants.items():
        timing[name] = _trace_us(mesh_device, lambda: project(x, w_l1, rs), calls, replays)
    for name, us in timing.items():
        logger.info(f"mhc decode ops traced {name}: {us} us a site (min of {replays} replays of {calls} calls)")
    result["site_us"] = timing
    logger.info(f"V41_MD_RESULT {json.dumps(result)}")
    out_path = os.environ.get("V41_MD_OUT")
    if out_path:
        with open(os.path.expanduser(out_path), "w") as f:
            json.dump(result, f, indent=1)
