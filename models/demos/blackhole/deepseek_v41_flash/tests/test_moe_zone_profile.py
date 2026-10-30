# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Zone-level device profile of dispatch + moe_compute at T rows per mesh row (env ZP_T, default 12), eager, one call per scenario.
Run with TT_METAL_DEVICE_PROFILER=1 and the kernels instrumented by dsv4-logs/roofline_s1/add_zones.py (point markers "MZ", data = id).
Scenarios: random top-6 routes (seeded) and n experts on device 0 (n = 1, 2, 4). Writes $ZP_OUT/zp.json (scenario -> program id range, busiest-device distinct experts)."""
import json
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.op_table_recorder import dev_id
from models.demos.blackhole.deepseek_v41_flash.tests.test_moe_stages import stages
from models.demos.blackhole.deepseek_v41_flash.tt.moe_block import DSV41MoEBlock
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import load_moe_layer

T = int(os.environ.get("ZP_T", "12"))
OUT = os.environ.get("ZP_OUT", "/tmp")


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 100_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(1500)
@torch.no_grad()
def test_zone_profile(mesh_device):
    md = mesh_device
    torch.manual_seed(0)
    blk = DSV41MoEBlock(md, load_moe_layer(2), batch_per_device=T, gate_bias_shift=0.0)
    blk.warmup()
    h = ttnn.from_torch(
        torch.randn(1, 1, T, 5120).to(torch.bfloat16),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    h_tok = ttnn.reshape(ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT), [T, 1, 1, 5120])
    rep_rows = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=tuple(md.shape))
    mk = lambda ids: ttnn.from_torch(
        ids.to(torch.int32).reshape(T * 4, 1, 1, 6),
        device=md,
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep_rows,
    )
    wts = ttnn.from_torch(
        torch.full((T * 4, 1, 1, 6), 0.25).to(torch.bfloat16),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep_rows,
    )
    g = torch.Generator().manual_seed(7)
    scen = {}
    rnd = torch.stack([torch.randperm(384, generator=g)[:6] for _ in range(T * 4)])
    scen["rand"] = rnd
    for n in (1, 2, 4):
        row = (list(range(n)) + [24, 36, 48, 60, 72, 84])[:6]
        scen[f"n{n}"] = torch.tensor(row).repeat(T * 4, 1)
    res = {}
    for name, ids in scen.items():
        # experts of device d: 12*d .. 12*d+11 (sequential mapping); distinct experts per device, per mesh row (a row's tokens go along the cluster axis)
        per_dev = {}
        for r in range(4):
            for e in set(ids[r * T : (r + 1) * T].flatten().tolist()):
                per_dev.setdefault(e // 12, set()).add(e)
        # moe_compute of mesh column c gets tokens of all 4 rows whose experts live in column c: devices of column c = 4 rows; expert owner device d -> distinct set over all rows
        allx = {}
        for e in set(ids.flatten().tolist()):
            allx.setdefault(e // 12, set()).add(e)
        tt_ids = mk(ids)
        stages(blk.decode, h_tok, wts, tt_ids, 1)  # warm this scenario's shapes
        ttnn.synchronize_device(md)
        if os.environ.get("TT_METAL_DEVICE_PROFILER"):
            ttnn.ReadDeviceProfiler(md)
        o0 = dev_id()
        out = stages(blk.decode, h_tok, wts, tt_ids, 1)
        o1 = dev_id()
        ttnn.synchronize_device(md)
        if (
            os.environ.get("ZP_SAVE") and name == "rand"
        ):  # combine output of every device, for bit-identity checks between configurations
            torch.save([ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)], os.environ["ZP_SAVE"])
        if os.environ.get("TT_METAL_DEVICE_PROFILER"):
            ttnn.ReadDeviceProfiler(md)
        res[name] = dict(
            op0=o0,
            op1=o1,
            max_distinct=max(len(v) for v in allx.values()),
            mean_distinct=sum(len(v) for v in allx.values()) / 32,
        )
        print("ZP", name, res[name], flush=True)
    json.dump(res, open(os.path.join(OUT, "zp.json"), "w"), indent=1)
