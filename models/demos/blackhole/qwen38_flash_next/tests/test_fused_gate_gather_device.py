# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device test of the GR gate + block-input gather as one program (``QWEN38_GATE_GATHER=1``, row_gather3): bitwise vs
``gate3.low_rank_gate3`` followed by ``row_gather2.row_gather`` on rows < rows (block, injection and the gathered hidden in the
8-core (gdn / qsa) and 5-core (moe) consumer layouts), outputs sentinel-filled first, repeated calls, a trace replayed with the
parity carried, and the traced time of both."""

import json
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests import test_fused_row_reduce_device as base
from models.demos.blackhole.qwen38_flash_next.tests.test_fused_row_reduce_device import device_params  # noqa: F401
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import row_gather2 as rg2
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import row_gather3 as rg3
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gr_read import gate3 as g3

TP = MESH_SHAPE[1]
CASES = [(1, 8), (2, 8), (1, 5), (2, 5)]
RESULTS = os.environ.get("ROW_REDUCE_RESULTS")
SENTINEL = float("nan")
ACTIVE = {"sentinel": True}  # off inside trace captures (a host write is not allowed there) and in the timing


def _cfg(cores):
    return ttnn.create_sharded_memory_config(
        (ttnn.TILE_SIZE, g3.LOCAL_HIDDEN * TP // cores),
        ttnn.CoreGrid(x=cores, y=1),
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _sentinel_allocate(monkeypatch):
    """Every program output starts as NaN (so an unwritten valid element cannot match by luck)."""

    real = fp.allocate

    def allocate(shape, dtype, layout, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        t = real(shape, dtype, layout, mesh, memory_config)
        if ACTIVE["sentinel"] and dtype == ttnn.bfloat16 and layout == ttnn.TILE_LAYOUT:
            fill = ttnn.full(
                list(shape), SENTINEL, dtype=dtype, layout=layout, device=mesh, memory_config=memory_config
            )
            ttnn.copy(fill, t)
            ttnn.deallocate(fill)
        return t

    monkeypatch.setattr(fp, "allocate", allocate)
    for mod in (g3, rg2, rg3):
        if hasattr(mod, "fp"):
            monkeypatch.setattr(mod.fp, "allocate", allocate)


def _mesh(mesh, x, dtype, replicate=False):
    mapper = ttnn.ReplicateTensorToMesh(mesh) if replicate else ttnn.ShardTensorToMesh(mesh, dim=0)
    return ttnn.from_torch(
        x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=mapper, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _host(mesh, t, rows):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:, :, :rows, :]


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [base.DEVICE_PARAMS], indirect=True)
def test_gate_gather_is_bitwise_gate3_plus_row_gather2(mesh_device, monkeypatch):
    mesh = mesh_device
    mesh.enable_program_cache()
    os.environ[rg3.ENV] = "1"
    rg3.prepare(mesh)
    _sentinel_allocate(monkeypatch)
    results = []
    for rows, cores in CASES:
        g = torch.Generator().manual_seed(11 + rows)
        partials = _mesh(mesh, torch.randn(TP, 1, rows, g3.PARTIAL_WIDTH, generator=g), ttnn.float32, replicate=False)
        # the gathered partials are replicated: every device holds the same [TP, 1, rows, W] (use device 0's shard as the value)
        gp = torch.randn(TP, 1, rows, g3.PARTIAL_WIDTH, generator=g)
        partials = ttnn.from_torch(
            gp,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        normalized = _mesh(mesh, torch.randn(TP, 1, rows, g3.FLAT_WIDTH, generator=g), ttnn.bfloat16)
        up = _mesh(mesh, (torch.randn(TP, 1, g3.PARTIAL_WIDTH, g3.FLAT_WIDTH, generator=g) * 0.1), ttnn.bfloat16)
        cfg = _cfg(cores)

        def chain():
            rg3.set_context(None)
            block, injection = g3.low_rank_gate3(partials, normalized, up)
            gathered = rg2.row_gather(block, cluster_axis=1, memory_config=cfg)
            return block, injection, gathered

        def fused():
            rg3.set_context(cfg)
            block, injection = g3.low_rank_gate3(partials, normalized, up)
            gathered = rg2.row_gather(block, cluster_axis=1, memory_config=cfg)
            return block, injection, gathered

        before = rg3.CALLS["fused"]
        ref = [_host(mesh, t, rows) for t in chain()]
        got = [_host(mesh, t, rows) for t in fused()]
        assert rg3.CALLS["fused"] == before + 1 and rg3.CALLS["handed"] >= 1, "the folded program was not taken"
        names = ("block", "injection", "gathered")
        report = {"rows": rows, "consumer_cores": cores}
        for name, a, b in zip(names, got, ref):
            report[f"{name}_mismatch"] = int((a.view(torch.int16) != b.view(torch.int16)).sum())
        rep = 0
        for _ in range(6):
            for a, b in zip([_host(mesh, t, rows) for t in fused()], ref):
                rep += int((a.view(torch.int16) != b.view(torch.int16)).sum())
        report["repeat_mismatch"] = rep
        fused()
        ACTIVE["sentinel"] = False
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            outs = [fused() for _ in range(5)]
        finally:
            ttnn.end_trace_capture(
                mesh, tid, cq_id=0
            )  # always closed: an exception inside must not leave the capture open
        bad = 0
        for _ in range(5):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            for o in outs:
                for a, b in zip([_host(mesh, t, rows) for t in o], ref):
                    bad += int((a.view(torch.int16) != b.view(torch.int16)).sum())
        ttnn.release_trace(mesh, tid)
        report["trace_replay_mismatch"] = bad

        def timed(fn):
            def call():
                block, injection, gathered = fn()
                ttnn.deallocate(block)
                ttnn.deallocate(injection)
                return gathered

            return base._traced_us(mesh, call)

        report["us_slope_and_mean12"] = {"gate3_plus_row_gather2": timed(chain), "gate_gather": timed(fused)}
        ACTIVE["sentinel"] = True
        print("GATE_GATHER_RESULT " + json.dumps(report))
        results.append(report)
        assert all(report[f"{n}_mismatch"] == 0 for n in names) and rep == 0 and bad == 0, report
    if RESULTS:
        with open(RESULTS, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
