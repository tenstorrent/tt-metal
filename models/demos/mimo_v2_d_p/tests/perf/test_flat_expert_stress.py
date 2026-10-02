# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hang hunt, op alone: FlatRoutedExpert back to back on every chip of the mesh (no other ops between launches unless
MIMO_FLAT_BETWEEN names some, as in test_flat_expert_mesh.py), cycling over MIMO_FSTRESS_SETS spiky count sets so the
active-expert schedule changes every launch. Every chip runs the op on every launch, so a hang that always lands on
the same chip points at that chip; one that wanders points at the kernel.

    MIMO_FSTRESS_ITERS (200000), MIMO_FSTRESS_SETS (16), MIMO_FSTRESS_SYNC (sync every N launches, 200), MIMO_FLAT_M,
    MIMO_FSTRESS_IMPL (cpp: the C++ op; py: the generic_op builder FlatExpert, whose MIMO_FL_* knobs then apply)
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping
from models.demos.mimo_v2_d_p.tests.mesh import MESH_PARAMS, mesh_id
from models.demos.mimo_v2_d_p.tests.unit.test_flat_expert_mesh import _between
from models.demos.mimo_v2_d_p.tt.flat_expert import FlatExpert, FlatRoutedExpert

H, I, E_GLOBAL = 4096, 2048, 256
ITERS = int(os.environ.get("MIMO_FSTRESS_ITERS", "200000"))
SETS = int(os.environ.get("MIMO_FSTRESS_SETS", "16"))
SYNC = int(os.environ.get("MIMO_FSTRESS_SYNC", "200"))


def _counts(g, n_dev, epc, m):
    """Per device: a few experts at (near) full capacity, half empty, the rest small (the model's routing shape)."""
    out = []
    for _ in range(n_dev):
        c = torch.randint(0, 200, (epc,), generator=g)
        c[torch.randperm(epc, generator=g)[: epc // 2]] = 0
        for e in torch.randperm(epc, generator=g)[: int(torch.randint(1, 4, (1,), generator=g))]:
            c[e] = m - int(torch.randint(0, 64, (1,), generator=g))
        out.append([int(v) for v in c])
    return out


@pytest.mark.timeout(86400)
@MESH_PARAMS
def test_flat_expert_stress(mesh_device, device_params):
    rows, cols = tuple(mesh_device.shape)
    n_dev = rows * cols
    epc = E_GLOBAL // n_dev
    m = int(os.environ.get("MIMO_FLAT_M", "2048"))
    table = ExpertMapping.create_global_expert_idx_table(
        experts_per_chip=epc, dispatch_group_size=rows, num_dispatch_groups=cols
    )
    gids = [[int(gl) for gl in table[c, r]] for r in range(rows) for c in range(cols)]
    torch.manual_seed(0)
    weights = [
        [(torch.randn(H, I) * 0.02, torch.randn(H, I) * 0.02, torch.randn(I, H) * 0.02) for _ in range(epc)]
        for _ in range(n_dev)
    ]
    impl = FlatExpert if os.environ.get("MIMO_FSTRESS_IMPL", "cpp") == "py" else FlatRoutedExpert
    fe = impl(mesh_device, weights, m=m, H=H, I=I, gids=gids, n_global=E_GLOBAL, pin=1)
    g = torch.Generator().manual_seed(int(os.environ.get("MIMO_FSTRESS_SEED", "0")))
    mesh_tensor = lambda ts, dt: ttnn.from_torch(
        torch.stack(ts),
        dtype=dt,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    sets, cap = [], 64
    for _ in range(SETS):
        local = _counts(g, n_dev, epc, m)
        offs = [[sum(-(-c // 32) * 32 for c in cl[:e]) for e in range(epc)] for cl in local]
        cap = max(cap, max(o[-1] + -(-cl[-1] // 32) * 32 for o, cl in zip(offs, local)) + 64)
        cnt_rows, reg_rows = [], []
        for dev in range(n_dev):
            c_ = torch.zeros(1, E_GLOBAL, dtype=torch.int32)
            r_ = torch.zeros(1, E_GLOBAL, dtype=torch.int32)
            for e, gl in enumerate(gids[dev]):
                c_[0, gl], r_[0, gl] = local[dev][e], offs[dev][e]
            cnt_rows.append(c_)
            reg_rows.append(r_)
        sets.append(
            (
                ttnn.reshape(mesh_tensor(cnt_rows, ttnn.uint32), (1, E_GLOBAL)),
                ttnn.reshape(mesh_tensor(reg_rows, ttnn.uint32), (1, E_GLOBAL)),
            )
        )
    x = ttnn.reshape(mesh_tensor([torch.randn(cap, H) for _ in range(n_dev)], ttnn.bfloat16), (cap, H))
    between = [op_ for op_ in os.environ.get("MIMO_FLAT_BETWEEN", "").split(",") if op_]
    t_start = t_last = time.perf_counter()
    for it in range(ITERS):
        counts, regions = sets[it % SETS]
        y = fe(x, counts, regions)
        ttnn.deallocate(y)
        for op_ in between:
            _between(op_, mesh_device)
        if (it + 1) % SYNC == 0:
            ttnn.synchronize_device(mesh_device)
            now = time.perf_counter()
            logger.info(
                f"FSTRESS {mesh_id(mesh_device)} launch {it + 1}/{ITERS}: {(now - t_last) / SYNC * 1e3:.2f} ms/launch, "
                f"elapsed {now - t_start:.0f} s"
            )
            t_last = now
    ttnn.synchronize_device(mesh_device)
    logger.info(f"FSTRESS done: {ITERS} launches, no hang, {time.perf_counter() - t_start:.0f} s")
