# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device test of QWEN38_GR_ROWS_FAST (gate3 with 8 x 32 chunked tiles, front3 with the u row multicast straight into the workers' tile
layout), rows 1 / 3 / 5 / 8: the flag off and on on the same inputs, valid rows bit for bit (the live-row form of the gate: the whole
32-row output, dead rows exact zeros), then the traced slope per call off vs on.
Run: QWEN38_FUSED_DEVICE_TEST=1 MESH_DEVICE=P150x8 TT_VISIBLE_DEVICES=... pytest this_file -s"""

import json
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import device_params  # noqa: F401
from models.demos.blackhole.qwen38_flash_next.tests import test_gr_front_device as F
from models.demos.blackhole.qwen38_flash_next.tests import test_gr_gate_device as G
from models.demos.blackhole.qwen38_flash_next.ttnn import decode_matmul as dm
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gr_read
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gr_read import gate3
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gr_recip_last import front3

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires the TP=8 mesh")
STEPS = int(os.environ.get("GRF_STEPS", "6"))
RESULTS = os.environ.get("GRF_RESULTS", "/tmp/grf_results.jsonl")
ENV = gr_read.ROWS_FAST_ENV
TP = MESH_SHAPE[1]


def flag(on):
    os.environ[ENV] = "1" if on else "0"


def bits(x):
    return x.view(torch.int16) if x.dtype == torch.bfloat16 else x.view(torch.int32)


def all_rows(outs):
    return [tuple(ttnn.to_torch(t).clone() for t in devs) for devs in zip(*(ttnn.get_device_tensors(t) for t in outs))]


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [G.DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("rows", [1, 3, 5, 8])
def test_gate_rows_fast(mesh_device, rows):
    mesh = mesh_device
    mesh.enable_program_cache()
    try:
        for step in range(STEPS):
            p, n, u = G.make_inputs(mesh, rows, 100 * rows + step)
            got = {}
            for on in (False, True):
                flag(on)
                o = gate3.low_rank_gate3(p, n, u, matmul="chain")
                ttnn.synchronize_device(mesh)
                got[on] = G.host_valid(o, rows)
                for t in o:
                    ttnn.deallocate(t)
            for d, (a, b) in enumerate(zip(got[False], got[True])):
                for what, x, y in zip(("block", "injection"), a, b):
                    assert torch.equal(bits(x), bits(y)), f"rows={rows} step {step} dev {d}: {what} differs"
            if 2 <= rows <= gate3.MAX_LIVE_ROWS:  # the verify's live-row form: the whole 32-row pages
                full = {}
                for on in (False, True):
                    flag(on)
                    o = gate3.low_rank_gate3(p, n, u, matmul="chain", out_rows=32)
                    ttnn.synchronize_device(mesh)
                    full[on] = all_rows(o)
                    for t in o:
                        ttnn.deallocate(t)
                for d, (a, b) in enumerate(zip(full[False], full[True])):
                    for what, x, y in zip(("block", "injection"), a, b):
                        assert torch.equal(
                            bits(x[:, :, :rows]), bits(y[:, :, :rows])
                        ), f"live rows={rows} step {step} dev {d}: {what} valid rows differ"
                        assert not y[
                            :, :, rows:
                        ].any(), f"live rows={rows} step {step} dev {d}: {what} dead rows not zero"
            for t in (p, n, u):
                ttnn.deallocate(t)
        logger.info(f"gate rows={rows}: flag off/on bit-identical over {STEPS} steps")
        p, n, u = G.make_inputs(mesh, rows, 7)
        times = {"off": [], "on": []}
        live_times = {"off": [], "on": []}
        for rep in range(3):
            for name, on in (("off", False), ("on", True)):
                flag(on)
                mesh.clear_program_cache()
                times[name].append(G.traced_us(mesh, lambda: gate3.low_rank_gate3(p, n, u, matmul="chain")))
                logger.info(f"gate rows={rows} rep {rep} {name}: slope/call {times[name][-1][0]} us")
                if 2 <= rows <= gate3.MAX_LIVE_ROWS:
                    mesh.clear_program_cache()
                    live_times[name].append(
                        G.traced_us(mesh, lambda: gate3.low_rank_gate3(p, n, u, matmul="chain", out_rows=32))
                    )
                    logger.info(f"gate live rows={rows} rep {rep} {name}: slope/call {live_times[name][-1][0]} us")
        with open(RESULTS, "a") as f:
            f.write(json.dumps(dict(what="gate", rows=rows, us=times, live_us=live_times)) + "\n")
    finally:
        os.environ.pop(ENV, None)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [F.DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("rows", [1, 3, 5, 8])
def test_front_rows_fast(mesh_device, rows):
    mesh = mesh_device
    mesh.enable_program_cache()
    try:
        for step in range(STEPS):
            res, gam, down = F.make_inputs(mesh, rows, 100 * rows + step)
            got = {}
            for on in (False, True):
                flag(on)
                o = front3.stats_recip_down_gather(res, gam, down)
                ttnn.synchronize_device(mesh)
                got[on] = F.valid_rows(o, rows)
                for t in o:
                    ttnn.deallocate(t)
            for d, (a, b) in enumerate(zip(got[False], got[True])):
                for what, x, y in zip(("stats", "normalized", "partials"), a, b):
                    if what != "stats":  # the gathered stats pages are not written by v3
                        assert torch.equal(bits(x), bits(y)), f"rows={rows} step {step} dev {d}: {what} differs"
            for t in (res, gam, down):
                ttnn.deallocate(t)
        logger.info(f"front rows={rows}: flag off/on bit-identical over {STEPS} steps")
        res, gam, down = F.make_inputs(mesh, rows, 7)
        times = {"off": [], "on": []}
        for rep in range(3):
            for name, on in (("off", False), ("on", True)):
                flag(on)
                mesh.clear_program_cache()
                times[name].append(F.traced_us(mesh, lambda: front3.stats_recip_down_gather(res, gam, down)))
                logger.info(f"front rows={rows} rep {rep} {name}: slope/call {times[name][-1][0]} us")
        with open(RESULTS, "a") as f:
            f.write(json.dumps(dict(what="front", rows=rows, us=times)) + "\n")
    finally:
        os.environ.pop(ENV, None)


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [F.DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("live", [3, 5])
def test_live_pipeline_rows_fast(mesh_device, live):
    """The verify's GR read (front v3 at live_rows + gate v3 at out_rows=32) with the flag off and on: the full 32-row outputs."""
    from models.demos.blackhole.qwen38_flash_next.tests import test_gr_verify_live_device as V

    mesh = mesh_device
    mesh.enable_program_cache()
    g = torch.Generator().manual_seed(11)
    up = ttnn.from_torch(
        (torch.randn(1, 1, F.PARTIAL, TP * F.FLAT, generator=g) * 0.05).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=dm.dram_sharded_weight_memory_config(mesh, F.PARTIAL, F.FLAT),
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(None, 3)),
    )
    try:
        for step in range(STEPS):
            _, gam, down = F.make_inputs(mesh, 1, 500 + step)
            res = V.residual32(mesh, live, 900 + 10 * live + step)
            got = {}
            for on in (False, True):
                flag(on)
                stats, norm, parts = front3.stats_recip_down_gather(res, gam, down, live_rows=live)
                ttnn.deallocate(stats)
                out = gate3.low_rank_gate3(parts, norm, up, matmul="chain", out_rows=32)
                got[on] = all_rows(out)
                for t in (*out, norm, parts):
                    ttnn.deallocate(t)
            for d, (a, b) in enumerate(zip(got[False], got[True])):
                for what, x, y in zip(("block", "injection"), a, b):
                    assert torch.equal(
                        bits(x[:, :, :live]), bits(y[:, :, :live])
                    ), f"live={live} step {step} dev {d}: {what}"
                    assert not y[:, :, live:].any(), f"live={live} step {step} dev {d}: {what} dead rows not zero"
            for t in (res, gam, down):
                ttnn.deallocate(t)
        logger.info(f"live={live}: front+gate flag off/on bit-identical over {STEPS} steps")
    finally:
        os.environ.pop(ENV, None)
