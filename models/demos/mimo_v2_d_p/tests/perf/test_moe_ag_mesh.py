# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The all-gather MoE block (tt/moe_ag.py MoeAgBlock) on QuietBox proxy meshes for the Galaxy (8 x 4) layout:
2x2 (the fast 2-row path), 4x1 (4 rows: the > 2-row send-back, no TP), 1x4 (no dispatch axis, 4-column TP all-reduce).
Random tokens / routing / expert outputs (y stands in for the expert: the block's data movement is what is checked):
gathered x, the route plan and the final [S, H] sums vs host references, plus device time per piece.

    MIMO_AG_MESH=4x1 MIMO_AG_SEQ=640,1280 scripts/run_safe_pytest.sh --profile \
        models/demos/mimo_v2_d_p/tests/perf/test_moe_ag_mesh.py

MIMO_AG_EPC=8: only 8 local experts per chip (the Galaxy's count; the other experts live nowhere) for the RoutePlan /
UntilizeActive limits.
"""

import os

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    fabric2d_device_params,
    torus_x_device_params,
    torus_y_device_params,
)
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, compute_constants
from models.demos.mimo_v2_d_p.tt.ffn import moe_capacity_factor
from models.demos.mimo_v2_d_p.tt.moe_ag import NONE, MoeAgBlock

try:
    from tracy import signpost
except ImportError:  # pragma: no cover
    signpost = lambda *a, **k: None

E, K, H = 256, 8, 4096
SEQS = [int(s) for s in os.environ.get("MIMO_AG_SEQ", "640").split(",")]
ITERS = int(os.environ.get("MIMO_CCL_ITERS", "2"))
_MESHES = {
    "2x2": pytest.param((2, 2), fabric2d_device_params(), id="2x2"),
    "4x1": pytest.param((4, 1), torus_y_device_params(), id="4x1"),
    "1x4": pytest.param((1, 4), torus_x_device_params(), id="1x4"),
}
MESHES = [_MESHES[m] for m in os.environ.get("MIMO_AG_MESH", "2x2,4x1,1x4").split(",")]


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _u32(t):
    return ttnn.to_torch(t).to(torch.int64) & 0xFFFFFFFF


def _timed(dev, tag, fn):
    out = fn()
    ttnn.synchronize_device(dev)
    for _ in range(ITERS):
        signpost(f"{tag}_start")
        out = fn()
        ttnn.synchronize_device(dev)
        signpost(f"{tag}_end")
    return out


@pytest.mark.parametrize("mesh_device, device_params", MESHES, indirect=["mesh_device", "device_params"])
def test_moe_ag_block(mesh_device, device_params):
    rows, cols = tuple(mesh_device.shape)
    n_dev = rows * cols
    epc_real = E // n_dev
    epc = int(os.environ.get("MIMO_AG_EPC", epc_real))
    table = ExpertMapping.create_global_expert_idx_table(
        experts_per_chip=epc_real, dispatch_group_size=rows, num_dispatch_groups=cols
    )
    gids = [[int(g) for g in table[c, r]][:epc] for r in range(rows) for c in range(cols)]
    mesh = f"{rows}x{cols}"
    gen = torch.Generator().manual_seed(3)
    for S in SEQS:
        T = rows * S
        _, _, buf_rows, _ = compute_constants(S, E, K, n_dev, rows, moe_capacity_factor(K, E, n_dev))
        blk = MoeAgBlock.get(
            mesh_device, chunk_size_per_chip=S, hidden=H, k=K, n_global=E, gids=gids, buf_rows=buf_rows
        )
        # per mesh row its S tokens (replicated over the columns)
        xt = torch.randn(rows, 1, S, H)
        it = torch.rand(T, E, generator=gen).argsort(-1)[:, :K]
        wt = torch.rand(T, K)
        row_shard = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(0, None))
        x = ttnn.from_torch(
            xt,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=row_shard,
        )
        idx = ttnn.from_torch(
            it.reshape(rows, 1, S, K).to(torch.int32),
            device=mesh_device,
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=row_shard,
        )
        w = ttnn.from_torch(
            wt.reshape(rows, 1, S, K),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=row_shard,
        )
        yt = torch.randn(rows, cols, buf_rows, H)
        y = ttnn.from_torch(
            yt,
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(0, 1)),
        )
        yq = [ttnn.to_torch(d).float().reshape(buf_rows, H) for d in ttnn.get_device_tensors(y)]

        x_rm = _timed(mesh_device, f"{mesh}_to_rm_S{S}", lambda: blk.to_rm(x))
        _timed(mesh_device, f"{mesh}_gather_S{S}", lambda: blk.gather(x_rm, idx, w))
        _timed(mesh_device, f"{mesh}_plan_S{S}", lambda: blk.plan())
        out = _timed(mesh_device, f"{mesh}_reduce_S{S}", lambda: blk.reduce(y))

        # gathered x: every chip of a column holds the column's T tokens in source-row order
        xall = xt.bfloat16().float().reshape(T, H)
        for d in ttnn.get_device_tensors(blk.gx):
            assert torch.equal(ttnn.to_torch(d).float().reshape(T, H), xall), "gathered x"
        # route plan + final sums vs the host
        ys = [_u32(d).reshape(T, K) for d in ttnn.get_device_tensors(blk.plan_op.y_slot)]
        counts = [_u32(d).reshape(-1) for d in ttnn.get_device_tensors(blk.plan_op.counts)]
        wq = wt.bfloat16().float()
        partial = torch.zeros(n_dev, T, H)
        for d in range(n_dev):
            loc = {g: l for l, g in enumerate(gids[d])}
            exp_counts = torch.zeros(E, dtype=torch.int64)
            for g in loc:
                exp_counts[g] = int((it == g).sum())
            assert torch.equal(counts[d], exp_counts), f"counts dev {d}"
            m = ys[d] != NONE
            assert int(m.sum()) == int(exp_counts.sum()), f"y_slot pairs dev {d}"
            gi, ki = m.nonzero(as_tuple=True)
            assert all(int(it[g, k]) in loc for g, k in zip(gi[:64].tolist(), ki[:64].tolist()))
            partial[d].index_add_(0, gi, wq[gi, ki, None] * yq[d][ys[d][gi, ki]].bfloat16().float())
        outs = [ttnn.to_torch(d).float().reshape(S, H) for d in ttnn.get_device_tensors(out)]
        worst = 1.0
        for d in range(n_dev):
            r = d // cols
            ref = partial[:, r * S : (r + 1) * S].sum(0)
            p = _pcc(outs[d], ref)
            worst = min(worst, p)
            assert p > 0.9999, f"final dev {d}: pcc {p}"
        used = sum(int(((c + 31) // 32 * 32).sum()) for c in counts[:1])
        print(
            f"AG_BLOCK_OK {mesh} S{S} T{T} epc{epc} buf_rows {buf_rows} (dev0 used {used}) min pcc {worst:.6f} "
            f"persistent {blk.buffers_mb():.1f} MB"
        )
        for t in (x, idx, w, y, out):
            ttnn.deallocate(t)
