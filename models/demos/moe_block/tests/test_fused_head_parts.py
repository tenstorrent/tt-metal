# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The fused MoE head's pieces on the LoudBox 2x4, one at a time (MOE_BLOCK_WORKLOG.md): the chunked matmul engine
(tt/lin2c.py) alone. Times need TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1
TT_METAL_PROFILER_CPP_POST_PROCESS=1."""

import os

import pytest
import torch
from loguru import logger
from ttnn.bringup.flat_routed_expert_ttnn.closeout_models import fabric_payload

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.moe_block.moe_block import ExpertActivation, act_host
from models.demos.moe_block.tests.test_tp_collectives import _time

R, C = 2, 4

# name: (K, N (gate / up pairs: 2 I), glu, GX)
LIN = {
    "glm_gu": (4096, 1024, "silu", 11),
    "glm_down": (512, 4096, None, 11),
    "k3_gu": (7168, 3072, "situ", 11),
    "k3_down": (1536, 7168, None, 11),
    "k3_gu_raw": (7168, 3072, None, 11),
    "glm_gu_raw": (4096, 1024, None, 11),
    "glm_gu_raw12": (4096, 1024, None, 12),
}


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((R, C), fabric2d_device_params(fabric_payload_size=fabric_payload(4096), l1_small_size=0))],
    indirect=True,
)
def test_lin2c(mesh_device, device_params):
    from models.demos.moe_block.tt.lin2c import Lin2c, glu_weight

    GY, MT = 10, int(os.environ.get("L2C_MT", "1"))
    MC = int(os.environ.get("L2C_MC", str(8 // MT)))
    rows = 32 * GY * MT * MC
    for name in os.environ.get("L2C_CASES", ",".join(LIN)).split(","):
        K, N, glu, GX = LIN[name]
        eng = Lin2c(mesh_device, cx0=0, GX=GX, GY=GY, MC=MC, K=K, N=N, glu=glu, MT=MT)
        g = torch.Generator().manual_seed(3)
        x = torch.randn(rows, K, generator=g)
        ws = [torch.randn(K, N, generator=g) * 0.02 for _ in range(R * C)]
        if glu:  # (W columns [gate | up] -> tile pairs)
            ws_dev = [glu_weight(w[:, : N // 2], w[:, N // 2 :]) for w in ws]
        else:
            ws_dev = ws
        wl = eng.upload(ws_dev)
        xt = ttnn.from_torch(
            x.view(1, 1, rows, K),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        us, y = _time(mesh_device, lambda: eng.run_alone(xt, wl))
        ys = [ttnn.to_torch(t).float().reshape(rows, -1) for t in ttnn.get_device_tensors(y)]
        ps = []
        for d in range(R * C):
            ref = x @ ws[d]
            if glu:
                ref = act_host(
                    ExpertActivation(glu, beta=4.0, linear_beta=25.0) if glu == "situ" else ExpertActivation("silu"),
                    ref[:, : N // 2],
                    ref[:, N // 2 :],
                )
            ps.append(pcc(ys[d], ref))
            if d == 0 and os.environ.get("L2C_DIAG"):  # (per chunk x column-group block)
                M_ = 32 * GY
                for m in range(MC):
                    row = [
                        round(
                            pcc(ys[0][m * M_ : (m + 1) * M_, c0 : c0 + 32], ref[m * M_ : (m + 1) * M_, c0 : c0 + 32]), 3
                        )
                        for c0 in range(0, ys[0].shape[1], 32)
                    ]
                    logger.info(f"L2C diag chunk {m}: per output tile column {row}")
        flops = 2 * rows * K * N
        logger.info(
            f"L2C {name} MT {MT} M {rows} K {K} N {N} glu {glu} GX {GX}: {us if us is None else round(us, 1)} us"
            + ("" if us is None else f" ({flops / us / 1e6:.0f} TFLOP/s)")
            + f", PCC min {min(ps):.5f}"
        )
        assert min(ps) > 0.999, ps
        ttnn.deallocate(y)


# name: (H, 2 I a chip (gate / up), glu, gate / up columns, down columns)
CHAIN = {
    "glm": (4096, 1024, "silu", 8, 4),
    "k3": (7168, 3072, "situ", 8, 4),
}
RS_DIM = {"glm": -2, "k3": -1}
LATENT = {"k3": 3584}  # (K3's latent width Hr)


def _routing_specs():
    from models.demos.moe_block.tt.fused_head import RoutingSpec

    return {"glm": RoutingSpec(A=12), "k3": RoutingSpec(A=16)}  # (the combine's accumulators: the close-out's)


class _LazyR(dict):
    def __missing__(self, k):
        self.update(_routing_specs())
        return dict.__getitem__(self, k)


ROUTING = _LazyR()


def _check_xbuf(head, xs, w_l, rows, name):
    """Dispatch prep's owned rows in the program: each chip's slots me S .. of the dispatch buffer [N S, W] vs its owned
    rows of x_full (bit for bit) or of x_full @ W_latent_down^T (K3: PCC)."""
    S = rows // C
    outs = [ttnn.to_torch(t).float().reshape(R * C * S, -1) for t in ttnn.get_device_tensors(head.outputs["xbuf"])]
    worst = 1.0
    for d in range(R * C):
        r, c = divmod(d, C)
        p = head.xg.chips[(r, c)]["rings"][0]["p"]
        xf = torch.cat([xs[r * C + cc] for cc in range(C)], dim=1)[p * S : (p + 1) * S]
        got = outs[d][d * S : (d + 1) * S]
        if w_l is None:
            exp = xf.to(torch.bfloat16).float()
            bad = (got != exp).any(dim=1).nonzero().flatten()
            assert bad.numel() == 0, f"{name} chip {d} owned rows: {bad.numel()} rows differ, first {bad[:8].tolist()}"
        else:
            worst = min(worst, pcc(got, xf @ w_l.T))
    if w_l is None:
        logger.info(f"HSE3 {name} dispatch rows (x): all {R * C} chips bit-exact")
    else:
        logger.info(f"HSE3 {name} dispatch rows (latent): PCC min {worst:.5f}")
        assert worst > 0.999, worst


def _check_route(head, rows, name):
    """The head's route words (dispatch prep in the program) vs combine_rf/dispatch_prep's route kernel run on the
    head's own joint rows, bit for bit."""
    from ttnn.bringup.combine_rf.dispatch_prep import dispatch_prep_prepare

    spec, S = ROUTER[name], rows // C
    mesh = head.mesh
    ref = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 1, S]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    xb = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 64]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    run = dispatch_prep_prepare(
        mesh, S=S, K=spec.K, EPC=head.routing.EPC, tokens_per_page=32, own_routing=True, copy=False
    )
    run(head.outputs["rows"], None, xb, ref)
    got = [
        ttnn.to_torch(t).to(torch.int64).reshape(-1) & 0xFFFFFFFF
        for t in ttnn.get_device_tensors(head.outputs["route"])
    ]
    exp = [ttnn.to_torch(t).to(torch.int64).reshape(-1) & 0xFFFFFFFF for t in ttnn.get_device_tensors(ref)]
    for ci, (g, e) in enumerate(zip(got, exp)):
        bad = (g != e).nonzero().flatten()
        assert bad.numel() == 0, f"{name} chip {ci} route words: {bad.numel()} differ, first {bad[:8].tolist()}"
    logger.info(f"HSE3 {name} route words: all {R * C} chips bit-exact vs dispatch_prep")


def _check_plan(head, rows, name):
    """combine_plan's per-chip tables from the head's routing exchange + planner vs its host model (plan_reference) on
    the device's own routing (every chip's joint rows), bit for bit."""
    from ttnn.bringup.combine_rf.combine_plan import plan_reference

    rt, spec = head.routing, ROUTER[name]
    K, S, N = spec.K, rows // C, R * C
    rows_t = [
        ttnn.to_torch(t).to(torch.int64).reshape(S, 2 * K) & 0xFFFF
        for t in ttnn.get_device_tensors(head.outputs["rows"])
    ]
    ids = torch.stack([r_[:, :K] for r_ in rows_t])
    wb = torch.stack([r_[:, K:] for r_ in rows_t])
    words = {
        k: [ttnn.to_torch(t).to(torch.int64).reshape(-1) & 0xFFFFFFFF for t in ttnn.get_device_tensors(v)]
        for k, v in rt.outs.items()
    }
    for ci in range(N):
        ch = (ci // C, ci % C)
        ref = plan_reference(
            ids,
            wb,
            ci,
            R,
            C,
            S,
            K,
            rt.EPC,
            N * rt.EPC,
            rt.A,
            rt.L,
            rt.nps[ch],
            rt.luts[ch],
            rt.rows_cap,
            token_major=True,
        )
        for k in ("counts", "regions", "token_index", "table", "plan"):
            got, exp = words[k][ci], torch.tensor(ref[k], dtype=torch.int64)
            if k in ("token_index", "table"):
                nw = min(ref["plan"][600], rt.rows_cap) * (2 if k == "table" else 1)
                got, exp = got[:nw], exp[:nw]
            bad = (got[: exp.numel()] != exp).nonzero().flatten()
            assert bad.numel() == 0, f"{name} chip {ci} {k}: {bad.numel()} words differ, first {bad[:8].tolist()}"
    logger.info(f"HSE3 {name} routing exchange + planner: all {N} chips bit-exact vs the host model")
    if rt.dbg is not None:  # (RA_DBG: the planner cores' phase stamps of the last call, us from the earliest)
        for ci, t in enumerate(ttnn.get_device_tensors(rt.dbg)):
            d = (ttnn.to_torch(t).to(torch.int64).reshape(-1, 32) & 0xFFFFFFFF)[: rt.M].tolist()
            t0 = min(r_[0] for r_ in d)
            logger.info(
                f"HSE3 {name} chip {ci} planner phases (us): "
                + " | ".join(" ".join(f"{(v - t0) / 1350:.1f}" for v in r_[:9]) for r_ in d)
            )
            logger.info(
                f"HSE3 {name} chip {ci} arrivals (us): " + " ".join(f"{(v - t0) / 1350:.1f}" for v in d[0][16:24])
            )


def _check_latent(head, xs, w_l, rows, name):
    """The owners' latent rows [rows / TP, Hr] vs x_full @ W_latent_down^T on the host."""
    G = C
    outs = [ttnn.to_torch(t).float().reshape(rows // G, -1) for t in ttnn.get_device_tensors(head.outputs["latent"])]
    ps = []
    for d in range(R * C):
        r, c = divmod(d, C)
        p = head.xg.chips[(r, c)]["rings"][0]["p"]
        xf = torch.cat([xs[r * C + cc] for cc in range(C)], dim=1)
        ref = xf[p * rows // G : (p + 1) * rows // G] @ w_l.T
        ps.append(pcc(outs[d], ref))
    logger.info(f"HSE3 {name} latent: PCC min {min(ps):.5f}")
    assert min(ps) > 0.999, ps


def _router_specs():
    from models.demos.moe_block.tt.fused_head import RouterSpec

    # (GLM-type: the logits ahead of the x gather; K3: after it)
    return {
        "glm": RouterSpec(E=288, K=8, route_scale=2.5),  # (the logits' priority: the head's choice)
        "k3": RouterSpec(E=448, K=16),
    }


class _Lazy(dict):
    def __missing__(self, k):
        self.update(_router_specs())
        return dict.__getitem__(self, k)


ROUTER = _Lazy()


def _check_routing(head, xs, w_r, b_r, rows_t, rows, name):
    """The owners' joint routing rows vs the host's fp32 routing (x as the device holds it, bf16)."""
    spec, G = ROUTER[name], C
    K = spec.K
    outs = [ttnn.to_torch(t).reshape(-1) for t in ttnn.get_device_tensors(rows_t)]
    agree, wp = [], []
    for d in range(R * C):
        r, c = divmod(d, C)
        p = head.xg.chips[(r, c)]["rings"][0]["p"]
        xf = torch.cat([xs[r * C + cc].to(torch.bfloat16).float() for cc in range(C)], dim=1)
        lg = xf @ w_r.T
        sc = torch.sigmoid(lg)
        n = rows // G
        tok = slice(p * n, (p + 1) * n)
        _, ids = torch.topk(sc[tok] + b_r, K, dim=-1)
        wts = sc[tok].gather(1, ids)
        wts = wts / wts.sum(-1, keepdim=True) * spec.route_scale
        u = outs[d].to(torch.int64).reshape(n, 2 * K) & 0xFFFF
        dids = u[:, :K]
        dw = (u[:, K:].to(torch.int32) << 16).view(torch.float32)
        hit = torch.stack([torch.isin(dids[t], ids[t]) for t in range(n)])
        agree.append(hit.float().mean().item())  # (the device's picks among the host's)
        same = hit.all(1)
        o_h, o_d = torch.sort(ids[same], 1)[1], torch.sort(dids[same], 1)[1]
        wp.append(pcc(torch.gather(wts[same], 1, o_h), torch.gather(dw[same], 1, o_d)))
    logger.info(
        f"HSE3 {name} routing: picks equal {100 * min(agree):.2f}% (worst chip), weights PCC {min(wp):.5f} "
        f"(rows with the same picks)"
    )
    assert min(agree) > 0.98 and min(wp) > 0.99, (
        agree,
        wp,
    )  # (the SE reduce-scatter: GLM's rows to their owners, K3's width)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((R, C), fabric2d_device_params(fabric_payload_size=fabric_payload(4096), l1_small_size=0))],
    indirect=True,
)
def test_lin2c_chain(mesh_device, device_params):
    """The shared expert as two chained lin2c engines in one program: gate / up + GLU on columns 0 .., h in DRAM,
    down on the next columns, chunk by chunk (the down engine's x senders wait for the gate / up row's notifies)."""
    from models.demos.moe_block.tt.lin2c import Lin2c, glu_weight

    GY, MT = 10, int(os.environ.get("L2C_MT", "1"))
    MC = int(os.environ.get("L2C_MC", str(8 // MT)))
    rows = 32 * GY * MT * MC
    logger.info(f"L2C grid {mesh_device.compute_with_storage_grid_size()}")
    for name in os.environ.get("L2C_CHAIN", ",".join(CHAIN)).split(","):
        H, N, glu, g1, g2 = CHAIN[name]
        if os.environ.get("L2C_SPLIT"):
            g1, g2 = map(int, os.environ["L2C_SPLIT"].split(":"))
        I = N // 2
        gu = Lin2c(mesh_device, cx0=0, GX=g1, GY=GY, MC=MC, K=H, N=N, glu=glu, MT=MT)
        dn = Lin2c(mesh_device, cx0=g1, GX=g2, GY=GY, MC=MC, K=I, N=H, xr_need=g1, MT=MT)
        g = torch.Generator().manual_seed(5)
        x = torch.randn(rows, H, generator=g)
        wg = [torch.randn(H, N, generator=g) * 0.02 for _ in range(R * C)]
        wd = [torch.randn(I, H, generator=g) * 0.03 for _ in range(R * C)]
        wl = gu.upload([glu_weight(w[:, :I], w[:, I:]) for w in wg])
        wld = dn.upload(wd)
        xt = ttnn.from_torch(
            x.view(1, 1, rows, H),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        us, y = _time(mesh_device, lambda: gu.run_chain(dn, xt, wl, wld))
        ys = [ttnn.to_torch(t).float().reshape(rows, -1) for t in ttnn.get_device_tensors(y)]
        act = ExpertActivation(glu, beta=4.0, linear_beta=25.0) if glu == "situ" else ExpertActivation("silu")
        ps = []
        for d in range(R * C):
            hh = x @ wg[d]
            ps.append(pcc(ys[d], act_host(act, hh[:, :I], hh[:, I:]) @ wd[d]))
        flops = 2 * rows * H * N + 2 * rows * I * H
        logger.info(
            f"L2C chain {name} MT {MT} M {rows} H {H} I {I} cols {g1} + {g2}: {us if us is None else round(us, 1)} us"
            + ("" if us is None else f" ({flops / us / 1e6:.0f} TFLOP/s)")
            + f", PCC min {min(ps):.5f}"
        )
        assert min(ps) > 0.999, ps
        ttnn.deallocate(y)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((R, C), fabric2d_device_params(fabric_payload_size=fabric_payload(4096), l1_small_size=0))],
    indirect=True,
)
def test_lin2d_ref(mesh_device, device_params):
    """(debug) combine_rf.lin2d itself at a lin2c case's shape, one call of 320 rows."""
    from ttnn.bringup.combine_rf.lin2d import lin2d_prepare, upload_stacked

    K, N, _, _ = LIN[os.environ.get("L2C_CASES", "glm_gu_raw")]
    run, lay = lin2d_prepare(mesh_device, M=320, K=K, N=N, fidelity=ttnn.MathFidelity.HiFi2)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(320, K, generator=g)
    w = torch.randn(K, N, generator=g) * 0.02
    wl = upload_stacked(mesh_device, w, lay)
    xt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    y = run(xt, wl)
    yt = ttnn.to_torch(ttnn.get_device_tensors(y)[0]).float()
    logger.info(f"L2C lin2d_ref K {K} N {N} pns {lay['pns']} NP {lay['NP']}: PCC {pcc(yt, x @ w):.5f}")


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((R, C), fabric2d_device_params(fabric_payload_size=fabric_payload(4096), l1_small_size=0))],
    indirect=True,
)
def test_head_xchain(mesh_device, device_params):
    """The fused head's x chain in one program a chip (tt/fused_head.py): the x all-gather over TP by the head's movers,
    chunk by chunk into the shared expert's gate / up + GLU, h, down -> y (the shared expert's partial a chip)."""
    from models.demos.moe_block.tt.fused_head import HeadLayout, HeadXChain
    from models.demos.moe_block.tt.lin2c import glu_weight

    rows = 2560
    for name in os.environ.get("L2C_CHAIN", ",".join(CHAIN)).split(","):
        H, N, glu, _, _ = CHAIN[name]
        I = N // 2
        probe = os.environ.get("HXC_PROBE", "")
        layout = HeadLayout(
            movers=os.environ.get("HXC_MOVERS", "last"),
            group=int(os.environ.get("HXC_GROUP", "8")),
            w_depth=int(os.environ.get("HXC_WD", "4")),
        )
        head = HeadXChain(mesh_device, rows=rows, H=H, I=I, glu=glu, probe=probe, layout=layout)
        g = torch.Generator().manual_seed(7)
        xs = [torch.randn(rows, H // C, generator=g) for _ in range(R * C)]
        wg = [torch.randn(H, N, generator=g) * 0.02 for _ in range(R * C)]
        wd = [torch.randn(I, H, generator=g) * 0.03 for _ in range(R * C)]
        wts = head.upload([glu_weight(w[:, :I], w[:, I:]) for w in wg], wd)
        xt = ttnn.from_torch(
            torch.stack(xs).view(R, C, rows, H // C),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(R, C)),
        )
        us, y = _time(mesh_device, lambda: head.run(xt, wts))
        ys = [ttnn.to_torch(t).float().reshape(rows, -1) for t in ttnn.get_device_tensors(y)]
        act = ExpertActivation(glu, beta=4.0, linear_beta=25.0) if glu == "situ" else ExpertActivation("silu")
        ps = []
        for d in range(R * C):
            r = d // C
            xf = torch.cat([xs[r * C + c] for c in range(C)], dim=1)
            hh = xf @ wg[d]
            ps.append(pcc(ys[d], act_host(act, hh[:, :I], hh[:, I:]) @ wd[d]))
        logger.info(
            f"HXC {name} {probe or 'fused'} rows {rows} H {H} I {I} layout {head.layout}: {us if us is None else round(us, 1)} us,"
            f" PCC min {min(ps):.5f}"
        )
        assert probe or min(ps) > 0.999, ps
        ttnn.deallocate(y)
        ttnn.deallocate(xt)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((R, C), fabric2d_device_params(fabric_payload_size=fabric_payload(4096), l1_small_size=0))],
    indirect=True,
)
def test_se_engine(mesh_device, device_params):
    """The shared expert engine alone (tt/se_engine.py: x in L1, every weight read once a call), x from DRAM."""
    from models.demos.moe_block.tt.lin2c import glu_weight
    from models.demos.moe_block.tt.se_engine import SeEngine

    rows, GY = 2560, 10
    for name in os.environ.get("L2C_CHAIN", ",".join(CHAIN)).split(","):
        H, N, glu, _, _ = CHAIN[name]
        I = N // 2
        GX = int(os.environ.get("SE_GX", "11"))
        if os.environ.get("SE_SHAPE"):  # (debug) rows,H,I,GX,GY
            rows, H, I, GX, GY = map(int, os.environ["SE_SHAPE"].split(","))
            N = 2 * I
        eng = SeEngine(
            mesh_device, rows=rows, H=H, I=I, glu=glu, GX=GX, GY=GY, KB=int(os.environ.get("SE_KB", "0")) or None
        )
        logger.info(
            f"SEE arena base {eng.base:#x} bytes {eng.arena_bytes:#x} end {eng.base + eng.arena_bytes:#x}; "
            + ", ".join(f"{k} {eng.base + v:#x}" for k, v in eng.off.items())
        )
        g = torch.Generator().manual_seed(5)
        x = torch.randn(rows, H, generator=g)
        wg = [torch.randn(H, N, generator=g) * 0.02 for _ in range(R * C)]
        wd = [torch.randn(I, H, generator=g) * 0.03 for _ in range(R * C)]
        w = eng.upload([glu_weight(t[:, :I], t[:, I:]) for t in wg], wd)
        xt = ttnn.from_torch(
            x.view(1, 1, rows, H),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        us, y = _time(mesh_device, lambda: eng.run_alone(xt, w))
        ys = [ttnn.to_torch(t).float().reshape(rows, -1) for t in ttnn.get_device_tensors(y)]
        act = ExpertActivation(glu, beta=4.0, linear_beta=25.0) if glu == "situ" else ExpertActivation("silu")
        ps = []
        for d in range(R * C):
            hh = x @ wg[d]
            ref = act_host(act, hh[:, :I], hh[:, I:]) @ wd[d]
            ps.append(pcc(ys[d], ref))
            if os.environ.get("SE_DIAG") and ps[-1] < 0.999:  # (debug) per row tile: per 32-column block PCC (x100)
                hd = act_host(act, hh[:, :I], hh[:, I:])
                for r0 in range(0, rows, 32):
                    row = [
                        int(100 * pcc(ys[d][r0 : r0 + 32, c0 : c0 + 32], ref[r0 : r0 + 32, c0 : c0 + 32]))
                        for c0 in range(0, H, 32)
                    ]
                    logger.info(f"SEE diag dev {d} row tile {r0 // 32}: {row}")
        db = eng.dram_bytes()
        flops = 2 * rows * H * N + 2 * rows * I * H
        logger.info(
            f"SEE {name} rows {rows} H {H} I {I} GX {eng.GX} KB {eng.KB} NPD {eng.NPD} arena {eng.arena_bytes // 1024} KB: "
            f"{us if us is None else round(us, 1)} us"
            + ("" if us is None else f" ({flops / us / 1e6:.0f} TFLOP/s)")
            + f", PCC min {min(ps):.5f}; DRAM a call: W {db['w'] / 1e6:.1f} MB (lin2c chain: "
            f"{(H * N + I * H) * 1088 / 1024 * 8 / 1e6:.1f}), x {db['x'] / 1e6:.1f}, y {db['y'] / 1e6:.1f}"
        )
        assert min(ps) > 0.999, ps
        ttnn.deallocate(y)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((R, C), fabric2d_device_params(fabric_payload_size=fabric_payload(4096), l1_small_size=0))],
    indirect=True,
)
def test_head_se(mesh_device, device_params):
    """The fused head's x chain v2 (tt/fused_head.py HeadSE): the x all-gather straight into the shared expert engine's
    L1, gate / up in shard arrival order, every weight read once."""
    from models.demos.moe_block.tt.fused_head import HeadSE
    from models.demos.moe_block.tt.lin2c import glu_weight

    rows = 2560
    for name in os.environ.get("L2C_CHAIN", ",".join(CHAIN)).split(","):
        H, N, glu, _, _ = CHAIN[name]
        I = N // 2
        head = HeadSE(mesh_device, rows=rows, H=H, I=I, glu=glu)
        g = torch.Generator().manual_seed(7)
        xs = [torch.randn(rows, H // C, generator=g) for _ in range(R * C)]
        wg = [torch.randn(H, N, generator=g) * 0.02 for _ in range(R * C)]
        wd = [torch.randn(I, H, generator=g) * 0.03 for _ in range(R * C)]
        w = head.upload([glu_weight(t[:, :I], t[:, I:]) for t in wg], wd)
        xt = ttnn.from_torch(
            torch.stack(xs).view(R, C, rows, H // C),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(R, C)),
        )
        us, y = _time(mesh_device, lambda: head.run(xt, w))
        ys = [ttnn.to_torch(t).float().reshape(rows, -1) for t in ttnn.get_device_tensors(y)]
        act = ExpertActivation(glu, beta=4.0, linear_beta=25.0) if glu == "situ" else ExpertActivation("silu")
        ps = []
        for d in range(R * C):
            r = d // C
            xf = torch.cat([xs[r * C + c] for c in range(C)], dim=1)
            hh = xf @ wg[d]
            ps.append(pcc(ys[d], act_host(act, hh[:, :I], hh[:, I:]) @ wd[d]))
        logger.info(
            f"HSE {name} rows {rows} H {H} I {I}: {us if us is None else round(us, 1)} us, PCC min {min(ps):.5f}"
            f" (orders: {[head.xg.order(c)[0] for c in sorted(head.xg.chips)][:4]})"
        )
        assert min(ps) > 0.999, ps
        ttnn.deallocate(y)
        ttnn.deallocate(xt)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((R, C), fabric2d_device_params(fabric_payload_size=fabric_payload(4096), l1_small_size=0))],
    indirect=True,
)
def test_se_engine3(mesh_device, device_params):
    """The shared expert engine v3 alone (tt/se_engine3.py: row-chunk-major, weights resident in L1), x from DRAM."""
    from models.demos.moe_block.tt.lin2c import glu_weight
    from models.demos.moe_block.tt.se_engine3 import SeEngine3

    rows, GX, GY, MT = int(os.environ.get("HXC_ROWS", "640")), 11, 10, int(os.environ.get("SE_MT", "1"))
    for name in os.environ.get("L2C_CHAIN", ",".join(CHAIN)).split(","):
        H, N, glu, _, _ = CHAIN[name]
        I = N // 2
        if os.environ.get("SE_SHAPE"):  # (debug) rows,H,I,GX,GY
            rows, H, I, GX, GY = map(int, os.environ["SE_SHAPE"].split(","))
            N = 2 * I
        eng = SeEngine3(
            mesh_device, rows=rows, H=H, I=I, glu=glu, GX=GX, GY=GY, MT=MT, KB=int(os.environ.get("SE_KB", "0")) or None
        )
        logger.info(
            f"SE3 {name}: chunks {eng.NCH} x {eng.CR * 32} rows, KB {eng.KB}, x families {eng.S0} + {eng.S1}, "
            f"passes gate/up {eng.NPG} down {eng.NPD}, W own {eng.wown_t} tiles, x own {eng.xown_t} tiles, "
            f"arena {eng.arena_bytes // 1024} KB"
        )
        g = torch.Generator().manual_seed(5)
        x = torch.randn(rows, H, generator=g)
        wg = [torch.randn(H, N, generator=g) * 0.02 for _ in range(R * C)]
        wd = [torch.randn(I, H, generator=g) * 0.03 for _ in range(R * C)]
        w = eng.upload([glu_weight(t[:, :I], t[:, I:]) for t in wg], wd)
        xt = ttnn.from_torch(
            x.view(1, 1, rows, H),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        us, y = _time(mesh_device, lambda: eng.run_alone(xt, w))
        ys = [ttnn.to_torch(t).float().reshape(rows, -1) for t in ttnn.get_device_tensors(y)]
        act = ExpertActivation(glu, beta=4.0, linear_beta=25.0) if glu == "situ" else ExpertActivation("silu")
        ps = []
        for d in range(R * C):
            hh = x @ wg[d]
            ps.append(pcc(ys[d], act_host(act, hh[:, :I], hh[:, I:]) @ wd[d]))
        flops = 2 * rows * H * N + 2 * rows * I * H
        logger.info(
            f"SE3 {name} rows {rows} H {H} I {I}: {us if us is None else round(us, 1)} us"
            + ("" if us is None else f" ({flops / us / 1e6:.0f} TFLOP/s)")
            + f", PCC min {min(ps):.5f}"
        )
        assert min(ps) > 0.999, ps
        ttnn.deallocate(y)
        ttnn.deallocate(eng.arena)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((R, C), fabric2d_device_params(fabric_payload_size=fabric_payload(4096), l1_small_size=0))],
    indirect=True,
)
def test_head_se3(mesh_device, device_params):
    """The fused head's x chain v3 (row-chunk-major, weights resident) (tt/fused_head.py HeadSE): the x all-gather straight into the shared expert engine's
    L1, gate / up in shard arrival order, every weight read once."""
    from models.demos.moe_block.tt.fused_head import HeadSE3
    from models.demos.moe_block.tt.lin2c import glu_weight

    rows = int(os.environ.get("HXC_ROWS", "640"))
    for name in os.environ.get("L2C_CHAIN", ",".join(CHAIN)).split(","):
        H, N, glu, _, _ = CHAIN[name]
        I = N // 2
        head = HeadSE3(
            mesh_device,
            rows=rows,
            H=H,
            I=I,
            glu=glu,
            MT=int(os.environ.get("SE_MT", "2")),
            KB=int(os.environ.get("SE_KB", "0")) or None,
            rs_dim=RS_DIM[name] if os.environ.get("SE_RS") else None,  # (+ down + SE reduce-scatter:)
            rs_mode=os.environ.get("SE_RS") or "fused",  # (fused: in this program; mrs: the ported op on h, the bar)
            router=ROUTER[name] if os.environ.get("SE_ROUTER") else None,  # (+ the router chain on its own cores)
            latent=LATENT.get(name) if os.environ.get("SE_LATENT") else None,  # (+ K3's latent down + all-to-all)
            routing=ROUTING[name] if os.environ.get("SE_ROUTING") else None,  # (+ the routing exchange + planner)
        )
        g = torch.Generator().manual_seed(7)
        xs = [torch.randn(rows, H // C, generator=g) for _ in range(R * C)]
        wg = [torch.randn(H, N, generator=g) * 0.02 for _ in range(R * C)]
        wd = [torch.randn(I, H, generator=g) * 0.03 for _ in range(R * C)]
        w_l = None
        if head.latent:
            w_l = torch.randn(head.latent, H, generator=g) * 0.02  # (the layer's latent down [Hr, H])
        w = head.upload(
            [glu_weight(t[:, :I], t[:, I:]) for t in wg], wd, w_latent=[w_l] * (R * C) if w_l is not None else None
        )
        if head.router is not None:
            spec = ROUTER[name]
            w_r, b_r = torch.randn(spec.E, H, generator=g) * 0.02, torch.randn(spec.E, generator=g) * 0.01
            head.upload_router(w_r, b_r)
        xt = ttnn.from_torch(
            torch.stack(xs).view(R, C, rows, H // C),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(R, C)),
        )
        us, y = _time(mesh_device, lambda: head.run(xt, w))
        if head.router is not None:
            _check_routing(head, xs, w_r, b_r, head.outputs["rows"], rows, name)
        if head.latent and "xbuf" not in head.outputs:
            _check_latent(head, xs, w_l, rows, name)
        if "xbuf" in head.outputs:
            _check_xbuf(head, xs, w_l, rows, name)
        if head.routing is not None:
            _check_plan(head, rows, name)
            if "route" in head.outputs:
                _check_route(head, rows, name)
        ys = [ttnn.to_torch(t).float().reshape(-1) for t in ttnn.get_device_tensors(y)]
        act = ExpertActivation(glu, beta=4.0, linear_beta=25.0) if glu == "situ" else ExpertActivation("silu")
        ps, refs = [], []
        for d in range(R * C):
            r = d // C
            xf = torch.cat([xs[r * C + c] for c in range(C)], dim=1)
            hh = xf @ wg[d]
            refs.append(act_host(act, hh[:, :I], hh[:, I:]) @ wd[d])
        for d in range(R * C):
            r, c = divmod(d, C)
            if head.rs_dim is None:
                ps.append(pcc(ys[d], refs[d].reshape(-1)))
            else:  # (chip (r, c): block c of the sum over its TP row)
                tot = sum(refs[r * C + cc] for cc in range(C))
                ps.append(pcc(ys[d], torch.chunk(tot, C, dim=head.rs_dim)[c].reshape(-1)))
        logger.info(
            f"HSE3 {name}{' +RS ' + head.rs_mode if head.rs_dim else ''} rows {rows} H {H} I {I} chunks {head.eng.NCH} x {head.eng.CR * 32}: "
            f"{us if us is None else round(us, 1)} us, PCC min {min(ps):.5f}"
        )
        assert os.environ.get("SE_PROBE") or min(ps) > 0.999, ps
        ttnn.deallocate(y)
        ttnn.deallocate(xt)
        if head.eng.arena.is_allocated():
            ttnn.deallocate(head.eng.arena)  # (the next model's arena needs the L1, unfragmented:)
        del head  # (the gather's global semaphores too)
        import gc

        from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter_program_descriptor as mrs

        mrs._MESH_CACHE.clear()  # (its staging / mailbox L1 tensors: the next arena wants the L1 unfragmented)
        gc.collect()
