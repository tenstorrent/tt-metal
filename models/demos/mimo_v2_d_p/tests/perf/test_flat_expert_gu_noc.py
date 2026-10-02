# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device time of the flat expert (generic_op builder FlatExpert) under the NOC0 hang fixes (see the program factory's
comment on se5_recv): MIMO_GUNOC_VARIANTS, ';'-separated env sets read at build time (default: the gate/up cores'
se5_recv on NOC0, then on NOC1), each built twice (alternating), each timed on every routing shape of MIMO_GUNOC_SHAPES
(spiky: _counts; balanced; light: 32 per expert; real_L1 / real_L5: the HF gate's counts of routing_counts_1280tok.json
scaled to MIMO_GUNOC_ASSIGN token-expert pairs, default a 4K chunk x top-8) at capacity MIMO_GUNOC_M (4096): median of
MIMO_GUNOC_CALLS calls (20) of the slowest chip's summed kernel time. MIMO_FL_XNOC is read at import (set it outside).

Needs TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1.
"""

import json
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping
from models.demos.mimo_v2_d_p.tests.mesh import MESH_PARAMS, mesh_id
from models.demos.mimo_v2_d_p.tests.perf.test_flat_expert_stress import _counts
from models.demos.mimo_v2_d_p.tt.flat_expert import FlatExpert

H, I, E_GLOBAL = 4096, 2048, 256


@pytest.mark.timeout(3600)
@MESH_PARAMS
def test_flat_expert_gu_noc(mesh_device, device_params):
    if not os.environ.get("TT_METAL_PROFILER_MID_RUN_DUMP"):
        pytest.skip("needs the in-process device profiler env")
    rows, cols = tuple(mesh_device.shape)
    n_dev = rows * cols
    epc = E_GLOBAL // n_dev
    table = ExpertMapping.create_global_expert_idx_table(
        experts_per_chip=epc, dispatch_group_size=rows, num_dispatch_groups=cols
    )
    gids = [[int(g) for g in table[c, r]] for r in range(rows) for c in range(cols)]
    torch.manual_seed(0)
    weights = [
        [(torch.randn(H, I) * 0.02, torch.randn(H, I) * 0.02, torch.randn(I, H) * 0.02) for _ in range(epc)]
        for _ in range(n_dev)
    ]
    m = int(os.environ.get("MIMO_GUNOC_M", "4096"))  # per-expert capacity (the model's max_tokens at a 4K chunk)
    chunk_assign = int(os.environ.get("MIMO_GUNOC_ASSIGN", str(4096 * 8)))  # token-expert pairs per dispatch group
    real = json.load(open(os.path.join(os.path.dirname(__file__), "routing_counts_1280tok.json")))
    gen = torch.Generator().manual_seed(5)

    def shape_counts(shape):
        """per device: its experts' token counts (local expert order)"""
        if shape == "spiky":
            return _counts(gen, n_dev, epc, m)
        if shape == "balanced":
            return [[min(m, chunk_assign // E_GLOBAL)] * epc for _ in range(n_dev)]
        if shape == "light":
            return [[32] * epc for _ in range(n_dev)]
        layer = shape.split("_")[1]  # real_L1 / real_L5: the HF gate's counts, scaled to the chunk
        scale = chunk_assign / sum(real[layer])
        return [[min(m, round(real[layer][g] * scale)) for g in gl] for gl in gids]

    mesh_tensor = lambda ts, dt: ttnn.from_torch(
        torch.stack(ts),
        dtype=dt,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    inputs, cap = {}, 64
    shapes = os.environ.get("MIMO_GUNOC_SHAPES", "spiky,balanced,real_L1,real_L5,light").split(",")
    for shape in shapes:
        local = shape_counts(shape)
        offs = [[sum(-(-c // 32) * 32 for c in cl[:e]) for e in range(epc)] for cl in local]
        cap = max(cap, max(o[-1] + -(-cl[-1] // 32) * 32 for o, cl in zip(offs, local)) + 64)
        cnt_rows, reg_rows = [], []
        for d in range(n_dev):
            c_ = torch.zeros(1, E_GLOBAL, dtype=torch.int32)
            r_ = torch.zeros(1, E_GLOBAL, dtype=torch.int32)
            for e, gl in enumerate(gids[d]):
                c_[0, gl], r_[0, gl] = local[d][e], offs[d][e]
            cnt_rows.append(c_)
            reg_rows.append(r_)
        inputs[shape] = (
            ttnn.reshape(mesh_tensor(cnt_rows, ttnn.uint32), (1, E_GLOBAL)),
            ttnn.reshape(mesh_tensor(reg_rows, ttnn.uint32), (1, E_GLOBAL)),
        )
        logger.info(f"GUNOC shape {shape}: tokens per device {[sum(cl) for cl in local]}, max {max(map(max, local))}")
    x = ttnn.reshape(mesh_tensor([torch.randn(cap, H) for _ in range(n_dev)], ttnn.bfloat16), (cap, H))
    calls = int(os.environ.get("MIMO_GUNOC_CALLS", "20"))
    variants = os.environ.get("MIMO_GUNOC_VARIANTS", "MIMO_FL_GU_NOC=0;MIMO_FL_GU_NOC=1").split(";")
    keys = {kv.split("=")[0] for v in variants for kv in v.split(",")}
    res = {}
    for noc in variants * 2:
        for k_ in keys:
            os.environ.pop(k_, None)
        for kv in noc.split(","):
            os.environ[kv.split("=")[0]] = kv.split("=")[1]
        fe = FlatExpert(mesh_device, weights, m=m, H=H, I=I, gids=gids, n_global=E_GLOBAL, pin=1)
        for shape in shapes:
            counts, regions = inputs[shape]
            for _ in range(3):
                ttnn.deallocate(fe(x, counts, regions))
            ttnn.synchronize_device(mesh_device)
            ts = []
            for _ in range(calls):
                ttnn.ReadDeviceProfiler(mesh_device)
                y = fe(x, counts, regions)
                ttnn.synchronize_device(mesh_device)
                ttnn.ReadDeviceProfiler(mesh_device)
                data = ttnn.get_latest_programs_perf_data()
                ts.append(
                    max(
                        sum(p.program_analyses_results["DEVICE KERNEL DURATION [ns]"].duration for p in progs) / 1e3
                        for progs in data.values()
                    )
                )
                ttnn.deallocate(y)
            ts.sort()
            res.setdefault((shape, noc), []).append(ts[len(ts) // 2])
            logger.info(f"GUNOC {shape} {noc}: median {ts[len(ts) // 2]:.1f} us, min {ts[0]:.1f}, max {ts[-1]:.1f}")
        del fe
    for shape in shapes:
        meds = {noc: sorted(res[(shape, noc)]) for noc in variants}
        base = sum(meds[variants[0]]) / len(meds[variants[0]])
        logger.info(
            f"GUNOC RESULT {mesh_id(mesh_device)} m {m} {shape}: "
            + ", ".join(f"{noc} {v} ({(sum(v) / len(v) / base - 1) * 100:+.1f}%)" for noc, v in meds.items())
        )
