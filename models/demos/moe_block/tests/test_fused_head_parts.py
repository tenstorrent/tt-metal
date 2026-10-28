# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The fused MoE head on the LoudBox 2x4 (MOE_BLOCK_WORKLOG.md): the shared expert engine alone (tt/se_engine3.py) and
the whole head (tt/fused_head.py HeadSE3) with its parts switched on by env (SE_RS, SE_ROUTER, SE_ROUTING, SE_LATENT);
every call on its own inputs, each checked against its own reference. Times need TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1
TT_METAL_PROFILER_CPP_POST_PROCESS=1."""

import os

import pytest
import torch
from loguru import logger
from ttnn.bringup.flat_routed_expert_ttnn.closeout_models import fabric_payload

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.moe_block.moe_block import ExpertActivation, act_host
from models.demos.moe_block.tests.test_tp_collectives import _calls, gen, ncalls

# (test-only: MOE_MESH "1x4" with MOE_FABRIC "torus_x" - one 2 x 2 square of the LoudBox as a TP ring of 4, the
# p150_x4_ring_1x4 descriptor and that square's TT_VISIBLE_DEVICES; default the 2 x 4 lines)
R, C = (int(v) for v in os.environ.get("MOE_MESH", "2x4").split("x"))
_FABRIC = {"torus_x": "FABRIC_2D_TORUS_X"}.get(os.environ.get("MOE_FABRIC", ""))


def _fab(**kw):
    p = fabric2d_device_params(**kw)
    if _FABRIC:
        p["fabric_config"] = getattr(ttnn.FabricConfig, _FABRIC)
    return p


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def _repl(mesh, t):
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _act(glu):
    return ExpertActivation(glu, beta=4.0, linear_beta=25.0) if glu == "situ" else ExpertActivation("silu")


def _se_ref(x, wg, wd, I, glu):
    hh = x @ wg
    return act_host(_act(glu), hh[:, :I], hh[:, I:]) @ wd


# name: (H, 2 I a chip (gate / up), glu, gate / up columns, down columns)
CHAIN = {
    "glm": (4096, 1024, "silu", 8, 4),
    "k3": (7168, 3072, "situ", 8, 4),
}
RS_DIM = {"glm": -2, "k3": -1}
LATENT = {"k3": 3584}  # (K3's latent width Hr)


def _routing_specs():
    from models.demos.moe_block.tt.fused_head import RoutingSpec

    prep = not os.environ.get("HSE3_NOPREP")  # (test-only: the routing without dispatch prep)
    tor = (_FABRIC == "FABRIC_2D_TORUS_X", False)  # (the dispatch's ports: a TP ring's wrap links too)
    return {"glm": RoutingSpec(A=12, prep=prep, torus=tor), "k3": RoutingSpec(A=16, prep=prep, torus=tor)}


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
            wrap=(rt.wrap_x, rt.wrap_y),  # (a ring's first hops: the planner's n_inj / n_ret and table hops)
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
                + " | ".join(
                    " ".join(f"{(v - t0) / 1350:.1f}" for v in r_[:9])
                    + f" [ER {r_[24]} n {r_[25]} hdr {(r_[26] - t0) / 1350:.1f} srcs {(r_[28] - t0) / 1350:.1f} rng {(r_[29] - t0) / 1350:.1f} zero {(r_[27] - t0) / 1350:.1f}]"
                    for r_ in d
                )
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
        if ps[-1] < 0.9995:  # (where: per source chip's column slice x row tile, the bad ones)
            LTc = ref.shape[1] // G
            bad = []
            for q in range(G):
                for rt in range(ref.shape[0] // 32):
                    o_, r_ = (
                        outs[d][rt * 32 : (rt + 1) * 32, q * LTc : (q + 1) * LTc],
                        ref[rt * 32 : (rt + 1) * 32, q * LTc : (q + 1) * LTc],
                    )
                    e = (o_ - r_).abs().max().item() / max(r_.abs().max().item(), 1e-6)
                    if e > 0.05:
                        cols = ((o_ - r_).abs() > 0.05 * r_.abs().max()).any(0).nonzero().flatten()
                        bad.append(
                            f"src{q}(pos) rt{rt} rel {e:.2f} cols {cols.min().item()}..{cols.max().item()} ({cols.numel()})"
                        )
            logger.info(f"HSE3 {name} latent chip {d} (p {p}): PCC {ps[-1]:.5f} bad {bad[:12]}")
    logger.info(f"HSE3 {name} latent: PCC min {min(ps):.5f}")
    assert min(ps) > 0.999, ps


def _dump_layout(head):
    """Actual L1 ranges a core type, from the built head (offsets + sizes), and the remote writers' highest target."""
    e, al = head.eng, lambda v: (v + 63) // 64 * 64  # noqa: E731
    off = sorted(e.off.items(), key=lambda kv: kv[1])
    ends = {}
    for i, (k, v) in enumerate(off):  # (a region ends where its size says, not at the next start)
        ends[k] = None
    sizes = dict(getattr(e, "rs_sizes", {}))
    sizes.update(
        xown=e.xown_t * 2048,
        wown=e.wown_t * e.wtb,
        xin=2 * e.MT * e.KB * 2048,
        words=256,
        acc=e.MT * (max(sum(c) for c in e.wg) + max(sum(e.wg, []))) * 4096,
        hown=e.MT * e.POMAX * 2048,
    )
    if getattr(e, "LT", None):
        sizes.update(lat=e.MT * e.LWMAX * 2048, latz=(e.shards - 1) * e.MT * e.LWMAX * 2048)
    rows = [(k, e.base + v, e.base + v + sizes[k]) for k, v in off if k in sizes]
    logger.info(
        "LAYOUT engine (base %d, arena %d B): " % (e.base, e.arena_bytes)
        + ", ".join(f"{k} {a}-{b}" for k, a, b in rows)
    )
    ovl_ok = {frozenset(p) for p in [("hold", "xin"), ("hold", "acc"), ("finb", "acc"), ("finb", "xin")]}
    for i, (k1, a1, b1) in enumerate(rows):
        for k2, a2, b2 in rows[i + 1 :]:
            if a1 < b2 and a2 < b1 and frozenset((k1, k2)) not in ovl_ok:
                logger.info(f"LAYOUT engine OVERLAP {k1} {a1}-{b1} x {k2} {a2}-{b2}")
    if e.base + e.arena_bytes < max(b for _, _, b in rows):
        logger.info(f"LAYOUT engine region past the arena end {e.base + e.arena_bytes}")
    rt = head.router
    if rt is not None:
        roff = sorted(rt.off.items(), key=lambda kv: kv[1])
        logger.info("LAYOUT router: " + ", ".join(f"{k} {rt.base + v}" for k, v in roff) + f" end {rt.base + rt.bytes}")
        logger.info(f"LAYOUT ports: pbuf {rt.pbase}-{rt.pbase + rt.pbuf_sub}")
    ra = head.routing
    if ra is not None:
        nf = max([len(v) for v in ra.fwd_plan()[0].values()] + [0])
        logger.info(
            f"LAYOUT routing: xbase {ra.xbase} fbase {ra.fbase} slots end {ra.fbase + nf * ra.PCAP} PCAP {ra.PCAP} planner regions {ra.pbases[:2]} step {ra.pl_step}"
        )
    logger.info(
        f"LAYOUT arena {e.base}-{e.base + e.arena_bytes}; mbox {int(head.mbox.buffer_address()) if getattr(head, 'mbox', None) is not None else None}"
    )


def _router_specs():
    from models.demos.moe_block.tt.fused_head import RouterSpec

    # (GLM-type: the logits ahead of the x gather; K3: after it) - the head's choices unless the test overrides them
    # (HSE3_ON_ENGINE / HSE3_LOGITS_FIRST / HSE3_W_GATE / HSE3_ORDERED = 0 | 1: the RouterSpec fields, for A/B runs)
    o = lambda k: None if os.environ.get(k) is None else bool(int(os.environ[k]))  # noqa: E731
    kw = dict(on_engine=o("HSE3_ON_ENGINE"), logits_first=o("HSE3_LOGITS_FIRST"), w_gate=o("HSE3_W_GATE"))
    kw["ordered"] = bool(o("HSE3_ORDERED"))
    return {
        "glm": RouterSpec(E=288, K=8, route_scale=2.5, **kw),
        # (test-only: HSE3_K3_E 896, the Galaxy's router; HSE3_K3_FOLD: candidates a planned expert - RouterConfig.fold)
        "k3": RouterSpec(
            E=int(os.environ.get("HSE3_K3_E", "448")), K=16, fold=int(os.environ.get("HSE3_K3_FOLD", "1")), **kw
        ),
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
        ids = ids // spec.fold
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
    # (E 896: sigmoid scores of random router weights sit near 0.5, the weights near 1 / K - their PCC is low-variance)
    assert min(agree) > 0.98 and min(wp) > (0.98 if spec.E >= 896 else 0.99), (
        agree,
        wp,
    )  # (the SE reduce-scatter: GLM's rows to their owners, K3's width)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((R, C), _fab(fabric_payload_size=fabric_payload(4096), l1_small_size=0))],
    indirect=True,
)
def test_se_engine3(mesh_device, device_params):
    """The shared expert engine v3 alone (tt/se_engine3.py: row-chunk-major, weights resident in L1), x from DRAM."""
    from ttnn.bringup.combine_rf.lin2d import glu_weight

    from models.demos.moe_block.tt.se_engine3 import SeEngine3

    rows, GX, GY, MT = int(os.environ.get("HXC_ROWS", "640")), 11, 10, int(os.environ.get("SE_MT", "1"))
    for name in os.environ.get("L2C_CHAIN", ",".join(CHAIN)).split(","):
        H, N, glu, _, _ = CHAIN[name]
        I = N // 2
        if os.environ.get("SE_SHAPE"):  # (debug) rows,H,I,GX,GY
            rows, H, I, GX, GY = map(int, os.environ["SE_SHAPE"].split(","))
            N = 2 * I
        eng = SeEngine3(
            mesh_device,
            rows=rows,
            H=H,
            I=I,
            glu=glu,
            GX=GX,
            GY=GY,
            MT=MT,
            KB=int(os.environ.get("SE_KB", "0")) or None,
            fidelity=getattr(ttnn.MathFidelity, os.environ.get("SE_FID", "HiFi2")),  # (a timing probe: LoFi)
        )
        logger.info(
            f"SE3 {name}: chunks {eng.NCH} x {eng.CR * 32} rows, KB {eng.KB}, x families {eng.S0} + {eng.S1}, "
            f"passes gate/up {eng.NPG} down {eng.NPD}, W own {eng.wown_t} tiles, x own {eng.xown_t} tiles, "
            f"arena {eng.arena_bytes // 1024} KB"
        )
        g = torch.Generator().manual_seed(5)
        wg = [torch.randn(H, N, generator=g) * 0.02 for _ in range(R * C)]
        wd = [torch.randn(I, H, generator=g) * 0.03 for _ in range(R * C)]
        w = eng.upload([glu_weight(t[:, :I], t[:, I:]) for t in wg], wd)
        xsk = [torch.randn(rows, H, generator=gen(5, k)) for k in range(ncalls())]  # (every call its own x)
        xts = [_repl(mesh_device, x.view(1, 1, rows, H)) for x in xsk]

        def check(k, y):
            if os.environ.get("SE_PROBE"):  # (a timing probe: no checks)
                return 1.0
            ys = [ttnn.to_torch(t).float().reshape(rows, -1) for t in ttnn.get_device_tensors(y)]
            ps = [pcc(ys[d], _se_ref(xsk[k], wg[d], wd[d], I, glu)) for d in range(R * C)]
            assert min(ps) > 0.999, (k, ps)
            return min(ps)

        us, ps = _calls(mesh_device, lambda k: eng.run_alone(xts[k], w), check)
        flops = 2 * rows * H * N + 2 * rows * I * H
        logger.info(
            f"SE3 {name} rows {rows} H {H} I {I}: {us if us is None else round(us, 1)} us"
            + ("" if us is None else f" ({flops / us / 1e6:.0f} TFLOP/s)")
            + f", PCC min {min(ps):.5f} ({len(ps)} calls, each its own x)"
        )
        for xt in xts:
            ttnn.deallocate(xt)
        ttnn.deallocate(eng.arena)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((R, C), _fab(fabric_payload_size=fabric_payload(4096), l1_small_size=0))],
    indirect=True,
)
def test_head_se3(mesh_device, device_params):
    """The fused head's x chain v3 (row-chunk-major, weights resident) (tt/fused_head.py HeadSE): the x all-gather straight into the shared expert engine's
    L1, gate / up in shard arrival order, every weight read once."""
    from ttnn.bringup.combine_rf.lin2d import glu_weight

    from models.demos.moe_block.tt.fused_head import HeadSE3

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
            GY=int(os.environ.get("HXC_GY", "10")),  # (test-only: other row counts - 1024 rows: 8 x MT 4)
            KB=int(os.environ.get("SE_KB", "0")) or None,
            rs_dim=RS_DIM[name] if os.environ.get("SE_RS") else None,  # (+ down + SE reduce-scatter:)
            rs_mode=os.environ.get("SE_RS") or "fused",  # (fused: in this program; mrs: the ported op on h, the bar)
            router=ROUTER[name] if os.environ.get("SE_ROUTER") else None,  # (+ the router chain on its own cores)
            latent=LATENT.get(name) if os.environ.get("SE_LATENT") else None,  # (+ K3's latent down + all-to-all)
            routing=ROUTING[name] if os.environ.get("SE_ROUTING") else None,  # (+ the routing exchange + planner)
        )
        g = torch.Generator().manual_seed(7)
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
            if spec.fold > 1:  # (one eligible candidate a group: the picks fold to distinct experts)
                from models.demos.moe_block.tests.test_moe_block import _fold_bias

                b_r = _fold_bias(b_r, spec.fold)
            head.upload_router(w_r, b_r)
        if os.environ.get("HSE3_LAYOUT"):  # (debug: every landing region's actual L1 range; overlaps flagged)
            _dump_layout(head)
        # (every call its own x - and so its own routing: each call checked against its own reference)
        soak = int(os.environ.get("HSE3_SOAK", "0"))  # (a soak: that many more calls over a pool of 8 inputs)
        xss = [
            [
                torch.randn(rows, H // C, generator=gen(int(os.environ.get("HSE3_SEED", "7")), k)) for _ in range(R * C)
            ]  # (HSE3_SEED: other inputs)
            for k in range(max(ncalls(), 8 if soak else 0))
        ]
        xts = [
            ttnn.from_torch(
                torch.stack(xs).view(R, C, rows, H // C),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(R, C)),
            )
            for xs in xss
        ]
        act = _act(glu)

        def check(k, y):
            xs = xss[k]
            if os.environ.get("SE_PROBE"):  # (a timing probe: no checks)
                return 1.0
            if head.router is not None:
                _check_routing(head, xs, w_r, b_r, head.outputs["rows"], rows, name)
            if head.latent and "xbuf" not in head.outputs:
                _check_latent(head, xs, w_l, rows, name)
            if "xbuf" in head.outputs and not os.environ.get(
                "HSE3_NOXBUF"
            ):  # (HSE3_NOXBUF: probes that skip the prep writes)
                _check_xbuf(head, xs, w_l, rows, name)
            if head.routing is not None:
                _check_plan(head, rows, name)
                if "route" in head.outputs:
                    _check_route(head, rows, name)
            ys = [ttnn.to_torch(t).float().reshape(-1) for t in ttnn.get_device_tensors(y)]
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
                    ref_d = torch.chunk(tot, C, dim=head.rs_dim)[c]
                    ps.append(pcc(ys[d], ref_d.reshape(-1)))
                    if not ps[-1] > 0.999:  # (where: per 32-row tile, finite and PCC)
                        yy = ys[d].reshape(ref_d.shape)
                        bad = [
                            (
                                t,
                                bool(torch.isfinite(yy[t * 32 : t * 32 + 32]).all()),
                                round(
                                    pcc(yy[t * 32 : t * 32 + 32].reshape(-1), ref_d[t * 32 : t * 32 + 32].reshape(-1)),
                                    3,
                                ),
                            )
                            for t in range(yy.shape[0] // 32)
                        ]
                        logger.info(f"HSE3 {name} chip {d} row tiles (finite, PCC): {bad}")
            logger.info(f"HSE3 {name} call {k}: PCC min {min(ps):.5f}")
            assert os.environ.get("SE_PROBE") or min(ps) > 0.999, (k, ps)
            return min(ps)

        us, pss = _calls(mesh_device, lambda k: head.run(xts[k], w), check)
        if (
            soak
        ):  # (each call its pool input; every HSE3_SOAK_CHECK-th (500) and the last checked against its own reference)
            every = int(os.environ.get("HSE3_SOAK_CHECK", "500"))
            import random as _rnd  # (HSE3_ORDER: input order; HSE3_SOAK_SYNC: a sync each call)

            order = list(range(len(xts)))
            if os.environ.get("HSE3_ORDER"):
                _rnd.Random(int(os.environ["HSE3_ORDER"])).shuffle(order)
            psync = bool(os.environ.get("HSE3_SOAK_SYNC"))
            for i in range(soak):
                y = head.run(xts[order[i % len(xts)]], w)
                if psync:
                    logger.info(f"HSE3 soak enqueued {i}")
                    ttnn.synchronize_device(mesh_device)
                    logger.info(f"HSE3 soak done {i}")
                if i % 50 == 49:  # (progress: where a hang lands)
                    ttnn.synchronize_device(mesh_device)
                    logger.info(f"HSE3 {name} soak progress: {i + 1}")
                if i % every == every - 1 or i == soak - 1:
                    ttnn.synchronize_device(mesh_device)
                    check(i % len(xts), y)
                    logger.info(f"HSE3 {name} soak: {i + 1} calls")
                ttnn.deallocate(y)
            ttnn.synchronize_device(mesh_device)
        if os.environ.get("SE_RTR_GATE_DBG"):  # PROBE: the engine cores' gate masks (word 60: 0x100 | arrived)
            for d, t in enumerate(ttnn.get_device_tensors(head.eng.ewords)):
                m = ttnn.to_torch(t).reshape(-1, 64)[:, 60].tolist()
                logger.info(f"GATE_DBG dev {d}: masks {sorted(set(hex(v) for v in m))} r {m[:12]}")
        if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":  # (traced: replays back to back, no launch skew)
            from models.demos.moe_block.tests.test_moe_block import replay_device_times

            ttnn.synchronize_device(mesh_device)
            tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            yt_ = head.run(xts[0], w)
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
            replay_device_times(mesh_device, tid, f"head {name}", batches=(4, 4))
            ttnn.release_trace(mesh_device, tid)
            ttnn.deallocate(yt_)
        if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":  # (the skew-corrected projection's map: each mesh row's
            # devices by TP position, next to the profile - tests/galaxy_projection.py)
            import json

            rmap = [[0] * C for _ in range(R)]
            for coord, ch in head.xg.chips.items():
                rmap[coord[0]][ch["rings"][0]["p"]] = mesh_device.get_device_id(ttnn.MeshCoordinate(*coord))
            os.makedirs("generated/profiler/.logs", exist_ok=True)
            json.dump(rmap, open("generated/profiler/.logs/head_map.json", "w"))
        ps = pss
        logger.info(
            f"HSE3 {name}{' +RS ' + head.rs_mode if head.rs_dim else ''} rows {rows} H {H} I {I} chunks {head.eng.NCH} x {head.eng.CR * 32}: "
            f"{us if us is None else round(us, 1)} us, PCC min {min(ps):.5f} ({len(ps)} calls, each its own x)"
        )
        for xt in xts:
            ttnn.deallocate(xt)
        if head.eng.arena.is_allocated():
            ttnn.deallocate(head.eng.arena)  # (the next model's arena needs the L1, unfragmented:)
        del head  # (the gather's global semaphores too)
        import gc

        from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter_program_descriptor as mrs

        mrs._MESH_CACHE.clear()  # (its staging / mailbox L1 tensors: the next arena wants the L1 unfragmented)
        gc.collect()


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((R, C), _fab(fabric_payload_size=fabric_payload(7168), l1_small_size=0))],
    indirect=True,
)
def test_router_vs_ops(mesh_device, device_params):
    """Diagnostic: the fused head's router picks and the ops head's router's (moe_block.Router: the same x, W_r, bias)
    vs the host's on bf16 W_r (what both devices hold): per chip the share of picks among the host's, and for the tokens
    either gets wrong the host's gap between its K-th and (K + 1)-th biased scores."""
    import dataclasses
    import gc

    from ttnn.bringup.combine_rf.lin2d import glu_weight

    from models.demos.moe_block.moe_block import GLM_53_FLASH, KIMI_K3, Geometry, MoEDtypes, Router
    from models.demos.moe_block.tt.fused_head import HeadSE3

    rows = 640
    # (GLM: K3's head leaves L1 the ops router's CBs then clash with - the block test compares K3's picks)
    for name in os.environ.get("L2C_CHAIN", "glm").split(","):
        H, N, glu, _, _ = CHAIN[name]
        I, spec = N // 2, ROUTER[name]
        K, G = spec.K, C
        g = torch.Generator().manual_seed(7)
        w_r, b_r = torch.randn(spec.E, H, generator=g) * 0.02, torch.randn(spec.E, generator=g) * 0.01
        xs = [torch.randn(rows, H // C, generator=gen(7, 0)) for _ in range(R * C)]
        xt = ttnn.from_torch(
            torch.stack(xs).view(R, C, rows, H // C),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(R, C)),
        )
        head = HeadSE3(
            mesh_device,
            rows=rows,
            H=H,
            I=I,
            glu=glu,
            rs_dim=RS_DIM[name],
            router=spec,
            latent=LATENT.get(name),
            routing=ROUTING[name],
        )
        wg = [torch.randn(H, N, generator=g) * 0.02 for _ in range(R * C)]
        wd = [torch.randn(I, H, generator=g) * 0.03 for _ in range(R * C)]
        w_l = [torch.randn(head.latent, H, generator=g) * 0.02] * (R * C) if head.latent else None
        w = head.upload([glu_weight(t[:, :I], t[:, I:]) for t in wg], wd, w_latent=w_l)
        head.upload_router(w_r, b_r)
        y = head.run(xt, w)
        ttnn.synchronize_device(mesh_device)
        fus = [ttnn.to_torch(t).reshape(-1) for t in ttnn.get_device_tensors(head.outputs["rows"])]
        pos = {d: head.xg.chips[divmod(d, C)]["rings"][0]["p"] for d in range(R * C)}
        for t in [y, w] + list(head.outputs.values()):
            ttnn.deallocate(t)
        head.free_arena()
        del head
        gc.collect()
        m = dataclasses.replace(GLM_53_FLASH if name == "glm" else KIMI_K3, num_experts=spec.E, top_k=K)
        gm = Geometry.of(mesh_device, m, rows * R, 2, zigzag=False)
        rt = Router(mesh_device, gm, m, {"weight": w_r, "bias": b_r}, MoEDtypes())
        lg = rt.logits(xt)
        own = ttnn.reduce_scatter(lg, dim=2, cluster_axis=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        orow = rt.top_k(own)
        ttnn.synchronize_device(mesh_device)
        ops = [ttnn.to_torch(t).reshape(-1) for t in ttnn.get_device_tensors(orow)]
        wq = w_r.to(torch.bfloat16).double()
        n = rows // G
        tot = {"fused": [], "ops": []}
        for d in range(R * C):
            r, c = divmod(d, C)
            xf = torch.cat([xs[r * C + cc].to(torch.bfloat16).double() for cc in range(C)], dim=1)
            bs = torch.sigmoid(xf @ wq.T) + b_r.double()
            out = []
            for tag, rows_, blk in (("fused", fus[d], pos[d]), ("ops", ops[d], c)):
                hb = bs[blk * n : (blk + 1) * n]
                srt, ids = torch.sort(hb, dim=-1, descending=True)
                gap = srt[:, K - 1] - srt[:, K]
                u = rows_.to(torch.int64).reshape(n, 2 * K) & 0xFFFF
                hit = torch.stack([torch.isin(u[t, :K], ids[t, :K]) for t in range(n)])
                bad = ~hit.all(1)
                gb = gap[bad]
                tot[tag].append(hit.float().mean().item())
                out.append(
                    f"{tag} {100 * hit.float().mean():.2f}% ({int(bad.sum())} tokens off, their host gap median "
                    f"{gb.median().item() if gb.numel() else 0:.1e} max {gb.max().item() if gb.numel() else 0:.1e}; "
                    f"all tokens' median {gap.median().item():.1e})"
                )
            logger.info(f"RTRVS {name} chip {d}: " + " | ".join(out))
        logger.info(
            f"RTRVS {name}: fused {100 * sum(tot['fused']) / len(tot['fused']):.2f}%, ops {100 * sum(tot['ops']) / len(tot['ops']):.2f}%"
        )
        for t in (lg, own, orow, xt):
            ttnn.deallocate(t)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((R, C), _fab(fabric_payload_size=fabric_payload(5120), l1_small_size=24576))],
    indirect=True,
)
def test_router_sqrtsoftplus(mesh_device, device_params):
    """The fused head's router with DeepSeek-V4.1's scores (RouterSpec score "sqrtsoftplus": sqrt(softplus), the fp32
    softplus; top 6, route scale 1.5) and its shared expert at LoFi (clamped SiLU 10), on its 2x2 (MOE_MESH=2x2: 1024
    rows a mesh row, 96 experts) vs float64 on the bf16 x and W_r the device holds: the picks of every token whose 6th /
    7th biased scores are apart by more than the logits' noise, the weights of the tokens with the same picks; the shared
    expert's owner rows by PCC."""
    from ttnn.bringup.combine_rf.lin2d import glu_weight

    from models.demos.moe_block.tt.fused_head import HeadSE3, RouterSpec

    rows, H, I, E, K, scale = R * 512, 5120, 2304 // C, 96, 6, 1.5
    spec = RouterSpec(E=E, K=K, route_scale=scale, score="sqrtsoftplus")
    gx = min(11, mesh_device.compute_with_storage_grid_size().x - 1)  # (as moe_block.FusedHead)
    head = HeadSE3(
        mesh_device,
        GX=gx,
        mover_col=gx,
        rows=rows,
        H=H,
        I=I,
        glu="clamped_silu",
        fidelity=ttnn.MathFidelity.LoFi,
        rs_dim=-2,
        router=spec,
    )
    g = torch.Generator().manual_seed(11)
    wg = [torch.randn(H, 2 * I, generator=g) * 0.02 for _ in range(R * C)]
    wd = [torch.randn(I, H, generator=g) * 0.03 for _ in range(R * C)]
    w = head.upload([glu_weight(t[:, :I], t[:, I:]) for t in wg], wd)
    w_r, b_r = torch.randn(E, H, generator=g) * 0.02, torch.randn(E, generator=g) * 0.1
    head.upload_router(w_r, b_r)
    G, n = C, rows // C
    act = ExpertActivation("clamped_silu", limit=10.0)
    wq = w_r.to(torch.bfloat16).double()
    worst = dict(clear=1.0, all=1.0, werr=0.0, se=1.0)
    for k in range(2):  # (two calls, each its own x)
        gk = gen(11, k)
        xs = [torch.randn(rows, H // C, generator=gk).to(torch.bfloat16) for _ in range(R * C)]
        xt = ttnn.from_torch(
            torch.stack(xs).view(R, C, rows, H // C),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(R, C)),
        )
        y = head.run(xt, w)
        ttnn.synchronize_device(mesh_device)
        got = [ttnn.to_torch(t).reshape(-1) for t in ttnn.get_device_tensors(head.outputs["rows"])]
        ys = [ttnn.to_torch(t).float().reshape(n, H) for t in ttnn.get_device_tensors(y)]
        for d in range(R * C):
            r, c = divmod(d, C)
            p = head.xg.chips[(r, c)]["rings"][0]["p"]
            xf = torch.cat([xs[r * C + cc].double() for cc in range(C)], dim=1)[p * n : (p + 1) * n]
            sc = torch.sqrt(torch.nn.functional.softplus(xf @ wq.T))
            srt, ids = torch.sort(sc + b_r.double(), dim=-1, descending=True)
            gap = srt[:, K - 1] - srt[:, K]
            wts = sc.gather(1, ids[:, :K])
            wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * scale
            u = got[d].to(torch.int64).reshape(n, 2 * K) & 0xFFFF
            dids, wb = u[:, :K], u[:, K:]
            dw = torch.where(wb >= 32768, wb - 65536, wb).to(torch.int16).view(torch.bfloat16).double()
            same = torch.tensor([torch.equal(torch.sort(dids[t])[0], torch.sort(ids[t, :K])[0]) for t in range(n)])
            clear = gap > 2e-3  # (the logits' noise: HiFi3 bf16 products, fp32 sums; the keys' 9 id bits)
            o_h, o_d = torch.sort(ids[same, :K], 1)[1], torch.sort(dids[same], 1)[1]
            werr = (torch.gather(wts[same], 1, o_h) - torch.gather(dw[same], 1, o_d)).abs().max().item()
            tot = sum(
                act_host(
                    act, *(torch.cat([xs[r * C + cc] for cc in range(C)], dim=1).float() @ wg[r * C + q]).split(I, 1)
                )
                @ wd[r * C + q]
                for q in range(C)
            )
            se = pcc(ys[d], torch.chunk(tot, C, dim=0)[c])
            logger.info(
                f"RTR_SSP call {k} chip {d} (p {p}): picks equal {100 * same.float().mean():.2f}% of tokens, "
                f"{100 * same[clear].float().mean():.2f}% of the {int(clear.sum())} clear of noise; weights max abs err "
                f"{werr:.2e} (same picks); shared expert PCC {se:.5f}"
            )
            worst = dict(
                clear=min(worst["clear"], same[clear].float().mean().item()),
                all=min(worst["all"], same.float().mean().item()),
                werr=max(worst["werr"], werr),
                se=min(worst["se"], se),
            )
        ttnn.deallocate(xt)
    logger.info(f"RTR_SSP worst: {worst}")
    assert worst["clear"] == 1.0, f"picks differ on tokens clear of noise: {worst}"
    assert worst["all"] > 0.99 and worst["werr"] < 4e-3, worst  # (bf16 weights ~0.25: half an ulp 5e-4)
    assert worst["se"] > 0.999, worst
