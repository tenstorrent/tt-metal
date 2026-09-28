# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FlatExpert (tt/flat_expert.py) alone on the mesh, in the MoE's formats: per device its 64 local experts (the EP
table's global ids), a row-major bf16 dispatch buffer, the routing's [1, 256] counts / regions rows. Checks every
active expert's rows against the quantized-weight reference on every device.

Counts are spiky like the model's routing (per device a few experts at the full capacity, half empty, the rest small).
Between launches the test runs other ops (MIMO_FLAT_BETWEEN, default a large matmul + SDPA + all-gather) and pins a
small L1 buffer (the next launch's arena then sits lower): a flat kernel that exits with NoC state the next program
trips on (a nonzero write transaction ID) hung exactly here, never back to back.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping
from models.demos.mimo_v2_d_p.tests.mesh import MESH_PARAMS
from models.demos.mimo_v2_d_p.tt.flat_expert import FlatExpert

H, I, E_GLOBAL = 4096, 2048, 256


def _counts(n_dev, epc, m):
    g = torch.Generator().manual_seed(3)
    out = []
    for dev in range(n_dev):
        c = torch.randint(0, 60, (epc,), generator=g)
        c[torch.randperm(epc, generator=g)[: epc // 2]] = 0  # half the experts empty
        c[dev % epc] = m  # one at the full capacity
        c[(dev + 7) % epc] = m - 3
        out.append([int(v) for v in c])
    return out


_BETWEEN = {}


def _between(name, mesh_device):
    """One op of the kind the decoder layer runs between two MoE calls (inputs built once)."""
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    mk = lambda *shape: ttnn.from_torch(
        torch.randn(*shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep
    )
    if name not in _BETWEEN:
        _BETWEEN[name] = {
            "matmul": lambda: (mk(1, 1, 1024, 4096), mk(1, 1, 4096, 4096)),
            "sdpa": lambda: (mk(1, 8, 1024, 192), mk(1, 8, 1024, 192), mk(1, 8, 1024, 192)),
            "rmsnorm": lambda: (mk(1, 1, 1024, 4096), mk(1, 1, 32, 4096)),
            "allgather": lambda: (mk(1, 1, 1024, 1024),),
        }[name]()
    t = _BETWEEN[name]
    if name == "matmul":
        out = ttnn.matmul(t[0], t[1])
    elif name == "sdpa":
        out = ttnn.transformer.scaled_dot_product_attention(t[0], t[1], t[2], is_causal=True)
    elif name == "rmsnorm":
        out = ttnn.rms_norm(t[0])
    elif name == "allgather":
        out = ttnn.all_gather(t[0], dim=-1, cluster_axis=1, topology=ttnn.Topology.Linear)
    ttnn.deallocate(out)


@pytest.mark.timeout(3600)
@MESH_PARAMS
def test_flat_expert_mesh(mesh_device, device_params):
    rows, cols = tuple(mesh_device.shape)
    n_dev = rows * cols
    epc = E_GLOBAL // n_dev
    m = int(os.environ.get("MIMO_FLAT_M", "2048"))
    table = ExpertMapping.create_global_expert_idx_table(
        experts_per_chip=epc, dispatch_group_size=rows, num_dispatch_groups=cols
    )
    gids = [[int(g) for g in table[c, r]] for r in range(rows) for c in range(cols)]
    local = _counts(n_dev, epc, m)
    torch.manual_seed(0)
    weights = [
        [(torch.randn(H, I) * 0.02, torch.randn(H, I) * 0.02, torch.randn(I, H) * 0.02) for _ in range(epc)]
        for _ in range(n_dev)
    ]
    fe = FlatExpert(mesh_device, weights, m=m, H=H, I=I, gids=gids, n_global=E_GLOBAL, pin=1)
    # per device: packed 32-row-aligned regions in local expert order, rows = the largest device's total
    offs = [[sum(-(-c // 32) * 32 for c in cl[:e]) for e in range(epc)] for cl in local]
    cap = max(o[-1] + -(-cl[-1] // 32) * 32 for o, cl in zip(offs, local)) + 64
    xs, cnt_rows, reg_rows = [], [], []
    for dev in range(n_dev):
        x_ = torch.zeros(cap, H)
        c_ = torch.zeros(1, E_GLOBAL, dtype=torch.int32)
        r_ = torch.zeros(1, E_GLOBAL, dtype=torch.int32)
        for e, g in enumerate(gids[dev]):
            x_[offs[dev][e] : offs[dev][e] + local[dev][e]] = torch.randn(local[dev][e], H)
            c_[0, g], r_[0, g] = local[dev][e], offs[dev][e]
        xs.append(x_)
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
    x = ttnn.reshape(mesh_tensor(xs, ttnn.bfloat16), (cap, H))
    counts = ttnn.reshape(mesh_tensor(cnt_rows, ttnn.uint32), (1, E_GLOBAL))
    regions = ttnn.reshape(mesh_tensor(reg_rows, ttnn.uint32), (1, E_GLOBAL))
    between = [op_ for op_ in os.environ.get("MIMO_FLAT_BETWEEN", "matmul,sdpa,allgather").split(",") if op_]
    pinned = []  # a small live L1 buffer appears after each launch: the next launch's arena sits lower
    y = None
    for it in range(int(os.environ.get("MIMO_FLAT_LAUNCHES", "3"))):
        if y is not None:
            ttnn.deallocate(y)
        y = fe(x, counts, regions)
        for op_ in between:
            _between(op_, mesh_device)
        pinned.append(
            ttnn.from_torch(
                torch.zeros(32 * 110 // 16, 32),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        )
    ttnn.synchronize_device(mesh_device)
    logger.info("all launches done")
    q = lambda w: ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()
    ys = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(y)]
    worst = 1.0
    for dev in range(n_dev):
        for e in range(epc):
            c = local[dev][e]
            if not c:
                continue
            rows_ = list(range(0, c, max(1, c // 64)))[:64] + [c - 1]  # (host reference cost)
            xe = xs[dev][offs[dev][e] : offs[dev][e] + c][rows_]
            Wg, Wu, Wd = (q(w_) for w_ in weights[dev][e])
            ref = (torch.nn.functional.silu(xe @ Wg) * (xe @ Wu)) @ Wd
            got = ys[dev].reshape(-1, H)[offs[dev][e] : offs[dev][e] + c][rows_]
            ok, pcc = comp_pcc(ref, got, 0.99)
            worst = min(worst, float(pcc))
            assert ok, (dev, e, c, pcc)
    logger.info(f"flat expert mesh: every active expert on {n_dev} devices, min PCC {worst:.5f}")
