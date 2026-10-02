# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host time of one flat_routed_expert call (program-cache hit): the Python call's wall time with the device idle
(synchronized before), median over launches, MiMo 64-expert plan at 2048 tokens per chip (indexed, row-major y).

    scripts/run_safe_pytest.sh models/demos/mimo_v2_d_p/tests/perf/test_flat_expert_host.py -s
"""

import os
import time

import pytest
from loguru import logger

import ttnn
from models.demos.mimo_v2_d_p.tests.mesh import MESH_PARAMS, mesh_id
from models.demos.mimo_v2_d_p.tests.unit.test_flat_expert_indexed_shapes import _call, _indexed_inputs, _op
from models.demos.mimo_v2_d_p.tests.unit.test_flat_routed_expert_op import CASES, _counts


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_flat_expert_host(device):
    case = CASES[0]
    op, gids = _op(device, case)
    d, _, _ = _indexed_inputs(device, case, gids, _counts(case[3], case[4], 1), 1)
    _call(op, d, 1, True).deallocate(True)  # compile
    host = []
    for _ in range(int(os.environ.get("MIMO_HOST_ITERS", "20"))):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        y = _call(op, d, 1, True)
        host.append((time.perf_counter() - t0) * 1e6)
        ttnn.synchronize_device(device)
        y.deallocate(True)
    s = sorted(host)
    logger.info(f"HOST flat_routed_expert call: median {s[len(s) // 2]:.0f} us, min {s[0]:.0f} us")


@pytest.mark.timeout(1800)
@MESH_PARAMS
def test_flat_expert_host_mesh(mesh_device, device_params):
    """Host time of one FlatRoutedExpert call on the mesh at the model's shape (m 4096, the real_L1 routing): idle
    device (synchronized before the call), and enqueue-only back to back (the device busy); a 1-op reference
    (an [4096, 4096] x [4096, 4096] matmul); MIMO_HOST_IMPL=py: the generic_op builder instead."""
    import torch

    from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping
    from models.demos.mimo_v2_d_p.tt.flat_expert import FlatExpert, FlatRoutedExpert

    H, I, E = 4096, 2048, 256
    rows, cols = tuple(mesh_device.shape)
    n_dev = rows * cols
    epc, m = E // n_dev, 4096
    table = ExpertMapping.create_global_expert_idx_table(
        experts_per_chip=epc, dispatch_group_size=rows, num_dispatch_groups=cols
    )
    gids = [[int(g) for g in table[c, r]] for r in range(rows) for c in range(cols)]
    torch.manual_seed(0)
    weights = [
        [(torch.randn(H, I) * 0.02, torch.randn(H, I) * 0.02, torch.randn(I, H) * 0.02) for _ in range(epc)]
        for _ in range(n_dev)
    ]
    impl = FlatExpert if os.environ.get("MIMO_HOST_IMPL") == "py" else FlatRoutedExpert
    fe = impl(mesh_device, weights, m=m, H=H, I=I, gids=gids, n_global=E, pin=1)
    import json

    real = json.load(open(os.path.join(os.path.dirname(__file__), "routing_counts_1280tok.json")))["L1"]
    scale = 4096 * 8 / sum(real)
    local = [[min(m, round(real[g] * scale)) for g in gl] for gl in gids]
    offs = [[sum(-(-c // 32) * 32 for c in cl[:e]) for e in range(epc)] for cl in local]
    cap = max(o[-1] + -(-cl[-1] // 32) * 32 for o, cl in zip(offs, local)) + 64
    cr, rr = [], []
    for d in range(n_dev):
        c_ = torch.zeros(1, E, dtype=torch.int32)
        r_ = torch.zeros(1, E, dtype=torch.int32)
        for e, gl in enumerate(gids[d]):
            c_[0, gl], r_[0, gl] = local[d][e], offs[d][e]
        cr.append(c_)
        rr.append(r_)
    mt = lambda ts, dt: ttnn.from_torch(
        torch.stack(ts),
        dtype=dt,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    counts, regions = ttnn.reshape(mt(cr, ttnn.uint32), (1, E)), ttnn.reshape(mt(rr, ttnn.uint32), (1, E))
    x = ttnn.reshape(mt([torch.randn(cap, H) for _ in range(n_dev)], ttnn.bfloat16), (cap, H))
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    a = ttnn.from_torch(
        torch.randn(1, 1, 4096, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep
    )

    def timed(fn, n=int(os.environ.get("MIMO_HOST_ITERS", "30"))):
        ttnn.deallocate(fn())
        idle = []
        for _ in range(n):
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            y = fn()
            idle.append((time.perf_counter() - t0) * 1e6)
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(y)
        t0 = time.perf_counter()
        ys = [fn() for _ in range(n)]
        busy = (time.perf_counter() - t0) * 1e6 / n
        ttnn.synchronize_device(mesh_device)
        for y in ys:
            ttnn.deallocate(y)
        idle.sort()
        return idle[n // 2], idle[0], busy

    for name, fn in (("flat_expert", lambda: fe(x, counts, regions)), ("matmul_ref", lambda: ttnn.matmul(a, a))):
        med, mn, busy = timed(fn)
        logger.info(
            f"HOSTMESH {mesh_id(mesh_device)} {impl.__name__ if name == 'flat_expert' else ''} {name}: idle median {med:.0f} us (min {mn:.0f}), back-to-back {busy:.0f} us/call"
        )
