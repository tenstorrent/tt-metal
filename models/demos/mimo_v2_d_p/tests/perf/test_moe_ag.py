# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""All-gather MoE block pieces (tt/moe_ag.py) on the 2x2 mesh: correctness vs host references + device time.

    scripts/run_safe_pytest.sh --profile models/demos/mimo_v2_d_p/tests/perf/test_moe_ag.py
"""

import os

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, compute_constants
from models.demos.mimo_v2_d_p.tests.mesh import MESH_PARAMS
from models.demos.mimo_v2_d_p.tt.ffn import moe_capacity_factor
from models.demos.mimo_v2_d_p.tt.moe_ag import NONE, RoutePlan, _dram
from models.demos.mimo_v2_d_p.tt.options import MiMoRuntimeOptions

try:
    from tracy import signpost
except ImportError:  # pragma: no cover
    signpost = lambda *a, **k: None

SEQS = [int(s) for s in os.environ.get("MIMO_CCL_SEQ", "640,2048").split(",")]
ITERS = int(os.environ.get("MIMO_CCL_ITERS", "3"))
E, K, H = 256, 8, 4096


def _timed(dev, tag, fn):
    out = fn()
    ttnn.synchronize_device(dev)
    for _ in range(ITERS):
        signpost(f"{tag}_start")
        fn()
        ttnn.synchronize_device(dev)
        signpost(f"{tag}_end")
    return out


def make_idx(T, gen, mode):
    if mode == "uniform":
        return torch.rand(T, E, generator=gen).argsort(-1)[:, :K]
    # skewed: a few hot experts (Zipf-like weights), distinct per token
    p = 1.0 / torch.arange(1, E + 1).float() ** 1.1
    p = p[torch.randperm(E, generator=gen)]
    return torch.multinomial(p.expand(T, E), K, replacement=False, generator=gen)


def mesh_gids(mesh_device, epc):
    rows, cols = tuple(mesh_device.shape)
    table = ExpertMapping.create_global_expert_idx_table(
        experts_per_chip=epc, dispatch_group_size=rows, num_dispatch_groups=cols
    )
    return [[int(g) for g in table[c, r]] for r in range(rows) for c in range(cols)]


def _u32(t):
    return ttnn.to_torch(t).to(torch.int64) & 0xFFFFFFFF


@MESH_PARAMS
@pytest.mark.parametrize("mode", ["uniform", "skewed"])
def test_route_plan(mesh_device, device_params, mode):
    rows, cols = tuple(mesh_device.shape)
    n_dev = rows * cols
    epc = E // n_dev
    gids = mesh_gids(mesh_device, epc)
    gen = torch.Generator().manual_seed(0)
    for S in SEQS:
        T = rows * S
        _, _, buf_rows, _ = compute_constants(
            S, E, K, n_dev, rows, moe_capacity_factor(K, E, n_dev, MiMoRuntimeOptions.from_env().moe_capacity)
        )
        idx = make_idx(T, gen, mode)
        idx_dev = ttnn.from_torch(
            idx.reshape(1, 1, T, K).to(torch.int32),
            device=mesh_device,
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        rp = RoutePlan(mesh_device, tokens=T, k=K, n_global=E, gids=gids, rows=buf_rows)
        _timed(mesh_device, f"route_plan_{mode}_S{S}", lambda: rp(idx_dev))
        if os.environ.get("MIMO_RP_STOP"):
            continue
        outs = [ttnn.get_device_tensors(t) for t in (rp.counts, rp.regions, rp.token_index, rp.y_slot)]
        for d in range(n_dev):
            lrow = torch.full((E,), NONE, dtype=torch.int64)
            for l, g in enumerate(gids[d]):
                lrow[g] = l
            c_ref, r_ref, t_ref, y_ref, used = RoutePlan.reference(idx, lrow, epc, buf_rows)
            assert used <= buf_rows, (used, buf_rows)
            c, r, t, y = (_u32(o[d]).reshape(-1) for o in outs)
            assert torch.equal(c, c_ref), f"counts dev {d}"
            assert torch.equal(r, r_ref), f"regions dev {d}"
            assert torch.equal(t[:used], t_ref[:used]), f"token_index dev {d}"
            assert torch.equal(y.reshape(T, K), y_ref), f"y_slot dev {d}"
        print(f"ROUTE_PLAN_OK {mode} S{S} rows used {used}/{buf_rows}")


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@MESH_PARAMS
def test_reduce_sendback(mesh_device, device_params):
    """route plan -> y (random rows; bfp8 TILE untilized on device) -> LocalReduce -> exchange (high_bw_all_gather
    over the rows + AddRows) -> TP all-reduce (high_bw_all_gather over the cols + AddRows), vs host sums."""
    from models.demos.mimo_v2_d_p.tt.moe_ag import AddRows, AddRowsTiled, LocalReduce, UntilizeActive, chip_info

    rows, cols = tuple(mesh_device.shape)
    assert rows == 2
    n_dev = rows * cols
    epc = E // n_dev
    gids = mesh_gids(mesh_device, epc)
    gen = torch.Generator().manual_seed(1)
    links = int(os.environ.get("MIMO_HBW_LINKS", "4"))
    for S in SEQS:
        T = rows * S
        _, _, buf_rows, _ = compute_constants(
            S, E, K, n_dev, rows, moe_capacity_factor(K, E, n_dev, MiMoRuntimeOptions.from_env().moe_capacity)
        )
        idx = make_idx(T, gen, "uniform")
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        idx_dev = ttnn.from_torch(
            idx.reshape(1, 1, T, K).to(torch.int32),
            device=mesh_device,
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        w = torch.rand(T, K)
        w_dev = ttnn.from_torch(
            w.reshape(1, 1, T, K),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
        y = torch.randn(n_dev, buf_rows, H)
        y_dev = ttnn.from_torch(
            y.reshape(rows, cols, buf_rows, H),
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(0, 1)),
        )
        yq = (
            ttnn.to_torch(
                y_dev, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(rows, cols), dims=(0, 1))
            )
            .float()
            .reshape(n_dev, buf_rows, H)
        )
        rp = RoutePlan(mesh_device, tokens=T, k=K, n_global=E, gids=gids, rows=buf_rows)
        info = chip_info(mesh_device, S)
        lr = LocalReduce(mesh_device, tokens=T, k=K, hidden=H, chunk_size_per_chip=S, split=True, info=info)
        ex = AddRows(mesh_device, n_rows=S, hidden=H, info=info)
        tp = AddRows(mesh_device, n_rows=S, hidden=H, info=info)
        g_sp = _dram(mesh_device, [1, 1, 2 * S, H])
        g_tp = _dram(mesh_device, [1, 1, 2 * S, H])

        _timed(mesh_device, f"route_plan_S{S}", lambda: rp(idx_dev))
        y_rm = _timed(mesh_device, f"untilize_y_S{S}", lambda: ttnn.to_layout(y_dev, ttnn.ROW_MAJOR_LAYOUT))
        if y_rm.dtype != ttnn.bfloat16:
            y_rm = ttnn.typecast(y_rm, ttnn.bfloat16)
        ua = UntilizeActive(mesh_device, rows=buf_rows, hidden=H, n_global=E, epc=epc, lmap=rp.lmap)
        y_rm = _timed(mesh_device, f"untilize_active_S{S}", lambda: ua(y_dev, rp.counts, rp.regions))
        _timed(mesh_device, f"local_reduce_S{S}", lambda: lr(y_rm, rp.y_slot, w_dev))
        own = [ttnn.to_torch(t).float().reshape(S, H) for t in ttnn.get_device_tensors(lr.own)]
        oth = [ttnn.to_torch(t).float().reshape(S, H) for t in ttnn.get_device_tensors(lr.other)]
        _timed(
            mesh_device,
            f"hbw_ag_sp_S{S}",
            lambda: ttnn.experimental.high_bw_all_gather(
                lr.other, dim=2, output_tensor=g_sp, cluster_axis=0, num_links=links
            ),
        )
        _timed(mesh_device, f"exchange_add_S{S}", lambda: ex(lr.own, g_sp, info_offset=True))
        _timed(
            mesh_device,
            f"hbw_ag_tp_S{S}",
            lambda: ttnn.experimental.high_bw_all_gather(
                ex.out, dim=2, output_tensor=g_tp, cluster_axis=1, num_links=links
            ),
        )
        _timed(mesh_device, f"tp_add_S{S}", lambda: tp(g_tp, g_tp, b_off=S))
        _timed(mesh_device, f"reduce_phase1_S{S}", lambda: lr.phase(y_rm, rp.y_slot, w_dev, 1))
        _timed(
            mesh_device,
            f"hbw_ag_sp2_S{S}",
            lambda: ttnn.experimental.high_bw_all_gather(
                lr.other, dim=2, output_tensor=g_sp, cluster_axis=0, num_links=links
            ),
        )
        _timed(mesh_device, f"reduce_phase2_S{S}", lambda: lr.phase(y_rm, rp.y_slot, w_dev, 2, peer=g_sp))
        own2 = [ttnn.to_torch(t).float().reshape(S, H) for t in ttnn.get_device_tensors(lr.own)]
        ex_out = [ttnn.to_torch(t).float().reshape(S, H) for t in ttnn.get_device_tensors(ex.out)]
        for d in range(n_dev):
            print(
                f"FUSED_SB S{S} dev {d}: pcc vs exchange add {_pcc(own2[d], ex_out[d]):.7f}"
                f" max abs diff {(own2[d] - ex_out[d]).abs().max():.3e}"
            )
            assert _pcc(own2[d], ex_out[d]) > 0.99999
        tpt = AddRowsTiled(mesh_device, n_rows=S, hidden=H)
        _timed(mesh_device, f"tp_add_tiled_S{S}", lambda: ttnn.deallocate(tpt(g_tp, g_tp, b_off=S)))
        fin_t = tpt(g_tp, g_tp, b_off=S)
        _timed(mesh_device, f"tilize_S{S}", lambda: ttnn.deallocate(ttnn.to_layout(tp.out, ttnn.TILE_LAYOUT)))

        # host reference
        ys_all = [(_u32(t).reshape(T, K)) for t in ttnn.get_device_tensors(rp.y_slot)]
        wq = w.bfloat16().float()
        partial = torch.zeros(n_dev, T, H)
        for d in range(n_dev):
            m = ys_all[d] != NONE
            gi, ki = m.nonzero(as_tuple=True)
            partial[d].index_add_(0, gi, wq[gi, ki, None] * yq[d][ys_all[d][gi, ki]].bfloat16().float())
        final = [ttnn.to_torch(t).float().reshape(S, H) for t in ttnn.get_device_tensors(tp.out)]
        final_t = [ttnn.to_torch(t).float().reshape(S, H) for t in ttnn.get_device_tensors(fin_t)]
        assert all(torch.equal(a, b) for a, b in zip(final, final_t)), "tiled add != row add"
        for d in range(n_dev):
            r = d // cols
            p_own, p_oth = _pcc(own[d], partial[d][r * S : (r + 1) * S]), _pcc(
                oth[d], partial[d][(1 - r) * S : (2 - r) * S]
            )
            ref = sum(partial[dd][r * S : (r + 1) * S] for dd in range(n_dev))
            p_fin = _pcc(final[d], ref)
            rel = float((final[d] - ref).norm() / ref.norm())
            print(f"REDUCE S{S} dev {d}: own {p_own:.6f} other {p_oth:.6f} final {p_fin:.6f} rel {rel:.2e}")
            assert min(p_own, p_oth, p_fin) > 0.9999, (p_own, p_oth, p_fin)


@MESH_PARAMS
def test_x_pages(mesh_device, device_params):
    """x TILE -> 2 KB-page row-major (UntilizeX) -> high_bw_all_gather of the [S * 4, 1024] view, vs to_layout +
    the 8 KB-page gather."""
    from models.demos.mimo_v2_d_p.tt.moe_ag import UntilizeX

    rows, cols = tuple(mesh_device.shape)
    links = int(os.environ.get("MIMO_HBW_LINKS", "4"))
    for S in SEQS:
        xt = torch.randn(rows, 1, S, H)
        x = ttnn.from_torch(
            xt,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(0, None)),
        )
        ux = UntilizeX(mesh_device, rows=S, hidden=H)
        _timed(mesh_device, f"untilize_x_pages_S{S}", lambda: ttnn.deallocate(ux(x)))
        _timed(mesh_device, f"to_layout_x_S{S}", lambda: ttnn.deallocate(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)))
        xr = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        _timed(mesh_device, f"reshape_x_S{S}", lambda: ttnn.reshape(xr, (1, 1, S * 4, 1024)))
        xrr = ttnn.reshape(xr, (1, 1, S * 4, 1024))
        print(f"RESHAPE same buffer: {xrr.buffer_address() == xr.buffer_address()}")
        xp = ux(x)
        assert all(
            torch.equal(ttnn.to_torch(a).float(), ttnn.to_torch(b).float())
            for a, b in zip(ttnn.get_device_tensors(xp), ttnn.get_device_tensors(xrr))
        ), "reshape != untilize_x"
        g = _dram(mesh_device, [1, 1, rows * S * 4, 1024])
        _timed(
            mesh_device,
            f"hbw_ag_x2k_S{S}",
            lambda: ttnn.experimental.high_bw_all_gather(xp, dim=2, output_tensor=g, cluster_axis=0, num_links=links),
        )
        ref = xt.bfloat16().float().reshape(rows * S * 4, 1024)
        for d in ttnn.get_device_tensors(g):
            assert torch.equal(ttnn.to_torch(d).float().reshape(-1, 1024), ref), "x pages gather mismatch"
        print(f"X_PAGES_OK S{S}")
