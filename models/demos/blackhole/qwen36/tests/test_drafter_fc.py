# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device: the drafter fc's 1D program (tt/mtp.py drafter_fc_progcfg) against the auto-configured program it replaces.

At the TP=8 Qwen3.8 shape ([rows, 2*dim] x [2*dim, dim/tp], bfloat8_b weight, tp_common.COMPUTE_HIFI2) and the
drafter's row counts (1 and 8 draft rows, 32 reseed rows), both programs run on the same inputs. The test logs how
many output elements differ and by how many bf16 steps, checks PCC, and logs the trace replay time of each program.

    MESH_DEVICE=P150x8 pytest models/demos/blackhole/qwen36/tests/test_drafter_fc.py
"""

import time

import pytest
import torch
from loguru import logger
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.test_factory import _resolve_mesh_shape

MESH = _resolve_mesh_shape(8)
DIM = 5120


def _parametrize(fn):
    params = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 64 << 20}
    fn = pytest.mark.parametrize("device_params", [params], indirect=True)(fn)
    return pytest.mark.parametrize("mesh_device", [pytest.param(MESH, id=f"{MESH[0]}x{MESH[1]}")], indirect=True)(fn)


def _replay_us(mesh, tid, n=50):
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - t0) / n * 1e6


def _per_device(t):
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


@torch.no_grad()
@_parametrize
def test_the_1d_fc_program_matches_the_auto_program(mesh_device):
    from models.demos.blackhole.qwen36.tt import tp_common as tpc
    from models.demos.blackhole.qwen36.tt.mtp import drafter_fc_progcfg

    nd = mesh_device.get_num_devices()
    g = torch.Generator().manual_seed(0)
    fc = tpc.shard_w(
        torch.randn(DIM, 2 * DIM, generator=g) * 0.02, mesh_device, -1, ttnn.DRAM_MEMORY_CONFIG, None, ttnn.bfloat8_b
    )
    pc = drafter_fc_progcfg(DIM, nd, mesh_device.compute_with_storage_grid_size().x)
    programs = {"auto": {}, "1d": {"program_config": pc}}

    def run(cat, tag):
        return ttnn.linear(
            cat,
            fc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=tpc.COMPUTE_HIFI2,
            **programs[tag],
        )

    for rows in (1, 8, 32):
        host = torch.randn(1, 1, rows, 2 * DIM, generator=g).to(torch.bfloat16)
        cat = ttnn.from_torch(
            host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        outs = {tag: run(cat, tag) for tag in programs}
        a, b = _per_device(outs["auto"]), _per_device(outs["1d"])
        differ, total, max_diff, max_val = 0, 0, 0.0, 0.0
        for d in range(nd):
            x, y = a[d].reshape(-1, a[d].shape[-1])[:rows], b[d].reshape(-1, b[d].shape[-1])[:rows]
            differ += int((x.view(torch.int16) != y.view(torch.int16)).sum())
            total += x.numel()
            max_diff = max(max_diff, float((x.float() - y.float()).abs().max()))
            max_val = max(max_val, float(x.float().abs().max()))
            ok, pcc = comp_pcc(x.float(), y.float(), 0.9999)
            assert ok, f"rows={rows} device {d}: 1d vs auto PCC {pcc}"
        logger.info(
            f"[DRAFTER_FC] rows={rows}: {differ} of {total} outputs differ between the programs; largest "
            f"|difference| {max_diff:.3g} (largest |output| {max_val:.3g})"
        )
        for t in outs.values():
            ttnn.deallocate(t)

        ttnn.synchronize_device(mesh_device)
        tids, held = {}, []
        with trace_allocation_tracker.corruptible_allocation_scope(mesh_device):
            for tag in programs:
                tids[tag] = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                held.append(run(cat, tag))
                ttnn.end_trace_capture(mesh_device, tids[tag], cq_id=0)
        us = {tag: _replay_us(mesh_device, tids[tag]) for tag in tids}
        logger.info(f"[DRAFTER_FC] rows={rows}: replay auto {us['auto']:.1f} us, 1d {us['1d']:.1f} us")
        for tag in tids:
            ttnn.release_trace(mesh_device, tids[tag])
        for t in held + [cat]:
            ttnn.deallocate(t)
    ttnn.deallocate(fc)
