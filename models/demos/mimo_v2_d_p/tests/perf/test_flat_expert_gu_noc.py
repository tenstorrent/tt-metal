# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device time of the flat expert (generic_op builder FlatExpert) under the NOC0 hang fixes (see the program factory's
comment on se5_recv): MIMO_GUNOC_VARIANTS, ';'-separated env sets (default: the gate/up cores' se5_recv on NOC0, on NOC1,
and the x relays' multicast on NOC1), each built twice; one spiky count set, median of MIMO_GUNOC_CALLS calls (20) of
the slowest chip's summed kernel time.

Needs TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping
from models.demos.mimo_v2_d_p.tests.mesh import MESH_PARAMS
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
    m = 2048
    table = ExpertMapping.create_global_expert_idx_table(
        experts_per_chip=epc, dispatch_group_size=rows, num_dispatch_groups=cols
    )
    gids = [[int(g) for g in table[c, r]] for r in range(rows) for c in range(cols)]
    torch.manual_seed(0)
    weights = [
        [(torch.randn(H, I) * 0.02, torch.randn(H, I) * 0.02, torch.randn(I, H) * 0.02) for _ in range(epc)]
        for _ in range(n_dev)
    ]
    local = _counts(torch.Generator().manual_seed(5), n_dev, epc, m)
    offs = [[sum(-(-c // 32) * 32 for c in cl[:e]) for e in range(epc)] for cl in local]
    cap = max(o[-1] + -(-cl[-1] // 32) * 32 for o, cl in zip(offs, local)) + 64
    cnt_rows, reg_rows = [], []
    for d in range(n_dev):
        c_ = torch.zeros(1, E_GLOBAL, dtype=torch.int32)
        r_ = torch.zeros(1, E_GLOBAL, dtype=torch.int32)
        for e, gl in enumerate(gids[d]):
            c_[0, gl], r_[0, gl] = local[d][e], offs[d][e]
        cnt_rows.append(c_)
        reg_rows.append(r_)
    mesh_tensor = lambda ts, dt: ttnn.from_torch(
        torch.stack(ts),
        dtype=dt,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    counts = ttnn.reshape(mesh_tensor(cnt_rows, ttnn.uint32), (1, E_GLOBAL))
    regions = ttnn.reshape(mesh_tensor(reg_rows, ttnn.uint32), (1, E_GLOBAL))
    x = ttnn.reshape(mesh_tensor([torch.randn(cap, H) for _ in range(n_dev)], ttnn.bfloat16), (cap, H))
    logger.info(f"GUNOC tokens per device {[sum(cl) for cl in local]}")
    calls = int(os.environ.get("MIMO_GUNOC_CALLS", "20"))
    variants = os.environ.get("MIMO_GUNOC_VARIANTS", "MIMO_FL_GU_NOC=0;MIMO_FL_GU_NOC=1;MIMO_FL_XNOC=1").split(";")
    keys = {kv.split("=")[0] for v in variants for kv in v.split(",")}
    res = {}
    for noc in variants * 2:
        for k_ in keys:
            os.environ.pop(k_, None)
        for kv in noc.split(","):
            os.environ[kv.split("=")[0]] = kv.split("=")[1]
        fe = FlatExpert(mesh_device, weights, m=m, H=H, I=I, gids=gids, n_global=E_GLOBAL, pin=1)
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
        res.setdefault(noc, []).append(ts[len(ts) // 2])
        logger.info(f"GUNOC {noc}: median {ts[len(ts) // 2]:.1f} us, min {ts[0]:.1f}, max {ts[-1]:.1f}")
        del fe
    logger.info(f"GUNOC summary (median per build): {res}")
