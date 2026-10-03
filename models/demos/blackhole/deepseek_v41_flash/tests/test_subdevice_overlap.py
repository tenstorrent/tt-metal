# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Feasibility spike: can sub-devices overlap the exact router with the shared expert inside the decode trace?

Modes (per rep = pre-op | overlap region (router + shared expert) | post-op; chain method (t(3 reps)-t(1 rep))/2):
  single   one trace, no manager (baseline A)
  seg2/3   same ops split into 2 / 3 trace segments, NO sub-devices (cost of a segment boundary, C)
  sd       3 segments, 2-sub-device manager loaded/cleared between segments on the host (B)
  whole    manager loaded BEFORE capture and kept for one single trace, never cleared inside (D)
"""

import time

import pytest
import torch

import ttnn
from models.common.modules.moe.tt_moe_gate_config import TTMoEGateConfig
from models.demos.blackhole.deepseek_v41_flash.reference.ref_layer import pcc
from models.demos.blackhole.deepseek_v41_flash.tt.moe_block import CONFIG_PATH
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import load_moe_layer
from models.demos.blackhole.deepseek_v41_flash.tt.router import ROUTED_GAIN, DSV41Gate
from models.demos.blackhole.deepseek_v41_flash.tt.shared_expert import DSV41SharedExpert

T = 4
ALT = [False]
D = 5120


class Prog:
    """Segmented trace program (SubDeviceTraceController-style): traces + host load/clear between them."""

    def __init__(self, md):
        self.md, self.steps, self.tid, self.keep, self.outs, self.whole = md, [], None, [], None, None

    def begin(self):
        self.tid = ttnn.begin_trace_capture(self.md, cq_id=0)

    def cut(self, action=None, payload=None):
        ttnn.end_trace_capture(self.md, self.tid, cq_id=0)
        self.steps.append(("trace", self.tid))
        if action == "load":
            self.md.load_sub_device_manager(payload)
        elif action == "clear":
            self.md.clear_loaded_sub_device_manager()
        if action:
            self.steps.append((action, payload))
        self.tid = ttnn.begin_trace_capture(self.md, cq_id=0)

    def end(self):
        ttnn.end_trace_capture(self.md, self.tid, cq_id=0)
        self.steps.append(("trace", self.tid))

    def replay(self):
        if self.whole is not None:
            self.md.load_sub_device_manager(self.whole)
        for kind, p in self.steps:
            if kind == "trace":
                ttnn.execute_trace(self.md, p, cq_id=0, blocking=False)
            elif kind == "load":
                self.md.load_sub_device_manager(p)
            else:
                self.md.clear_loaded_sub_device_manager()
        if self.whole is not None:
            self.md.clear_loaded_sub_device_manager()

    def release(self):
        loaded = self.whole is not None
        if loaded:
            self.md.load_sub_device_manager(self.whole)
        for kind, p in self.steps:
            if kind == "trace":
                ttnn.release_trace(self.md, p)
            elif kind == "load":
                self.md.load_sub_device_manager(p)
                loaded = True
            else:
                self.md.clear_loaded_sub_device_manager()
                loaded = False
        if loaded:
            self.md.clear_loaded_sub_device_manager()
        self.steps, self.keep = [], []


class Ctx:
    """Holds op-confinement state: which ops accept sub_core_grids / sub_device_id."""

    def __init__(self):
        self.on = False
        self.cores = None
        self.sd = None
        self.unsupported = set()
        self.supported = set()
        self.fatal = set()

    def op(self, name, fn, *a, sd=False, confine=True, **kw):
        if not (self.on and confine) or name in self.unsupported:
            return fn(*a, **kw)
        k = dict(sub_device_id=self.sd) if sd else dict(sub_core_grids=self.cores)
        try:
            r = fn(*a, **kw, **k)
            self.supported.add(name)
            return r
        except Exception as e:  # noqa: BLE001
            self.unsupported.add(name)
            print(f"SDO cannot confine {name}: {str(e).splitlines()[0][:140]}", flush=True)
            try:
                return fn(
                    *a, **kw
                )  # unconfined under a loaded 2-sub-device manager: works for some ops, fatal for others
            except Exception as e2:  # noqa: BLE001
                self.fatal.add(name)
                print(
                    f"SDO {name} UNCONFINED ALSO FATAL under loaded manager: {' '.join(str(e2).split())[:100]}",
                    flush=True,
                )
                raise RuntimeError(f"UNCONFINABLE {name}")


def router_steps(gate, h, cx, keep):
    """Generator mirroring DSV41Gate._forward_exact_fast, one op per step; final (weights, indices) via StopIteration."""
    E, k = gate._E, gate.k
    op = cx.op
    cg = ttnn.CoreGrid(y=2, x=8)
    logits = op(
        "r.matmul",
        ttnn.matmul,
        h,
        gate._w_exact,
        sd=True,
        compute_kernel_config=gate._ckc_exact,
        dtype=ttnn.float32,
        core_grid=cg,
    )
    keep.append(logits)
    yield
    sp = op("r.softplus", ttnn.softplus, logits, beta=1.0, threshold=20.0)
    keep.append(sp)
    yield
    score = op("r.sqrt", ttnn.sqrt, sp)
    keep.append(score)
    yield
    rank = op("r.add_bias", ttnn.add, score, gate._bias_exact)
    keep.append(rank)
    return score, rank


def router_stage2(gate, score, rank, cx, keep):
    """topk + normalisation: topk/sum/to_layout cannot be confined (fatal under a loaded 2-sub-device manager), so
    this runs on the full grid after the manager is cleared."""
    E, k = gate._E, gate.k
    op = cx.op
    _, idx = op("r.topk", ttnn.topk, rank, k=k, dim=-1, largest=True, sorted=True)
    keep.append(idx)
    idf = op("r.typecast_f32", ttnn.typecast, idx, ttnn.float32)
    idr = ttnn.reshape(idf, [T, 1, k, 1])
    onehot = op("r.eq", ttnn.eq, idr, gate._arange)
    keep += [idf, idr, onehot]
    sc = ttnn.reshape(score, [T, 1, 1, E])
    prod = op("r.mul", ttnn.multiply, onehot, sc)
    keep.append(prod)
    sel = op("r.sum_E", ttnn.sum, prod, dim=-1, keepdim=True)
    keep.append(sel)
    s2 = op("r.sum_k", ttnn.sum, sel, dim=2, keepdim=True)
    denom = op("r.add_eps", ttnn.add, s2, 1e-20)
    keep += [s2, denom]
    q = op("r.div", ttnn.div, sel, denom)
    q = op("r.mul_scale", ttnn.multiply, q, gate.scaling_factor / ROUTED_GAIN)
    w = op("r.typecast_bf16", ttnn.typecast, q, ttnn.bfloat16)
    keep += [q, w]
    wl = op("r.to_layout_w", ttnn.to_layout, w, ttnn.ROW_MAJOR_LAYOUT)
    weights = ttnn.reshape(wl, (T, 1, 1, k))
    iu = op("r.typecast_u16", ttnn.typecast, idx, ttnn.uint16)
    il = op("r.to_layout_i", ttnn.to_layout, iu, ttnn.ROW_MAJOR_LAYOUT)
    indices = ttnn.view(il, (T, 1, 1, k))
    keep += [wl, weights, iu, il, indices]
    return weights, indices


def shared_steps(se, h, cx, keep):
    inter, lim = se.inter, se.limit
    op = cx.op
    gu = op(
        "s.linear_gu",
        ttnn.linear,
        h,
        se.w01,
        sd=True,
        dtype=ttnn.float32,
        compute_kernel_config=se.ckc,
        core_grid=se.grid,
    )
    keep.append(gu)
    yield
    g = op("s.slice_gate", ttnn.slice, gu, [0, 0, 0, 0], [1, 1, gu.shape[2], inter])
    u = op("s.slice_up", ttnn.slice, gu, [0, 0, 0, inter], [1, 1, gu.shape[2], 2 * inter])
    keep += [g, u]
    yield
    if ALT[
        0
    ]:  # confinable formulation (ttnn.minimum / ttnn.clamp take no sub_core_grids): min(g,L)=g-relu(g-L); clamp(u)=relu_max(u+L,2L)-L
        t1 = op("s.add1", ttnn.add, g, -lim)
        t2 = op("s.relu_max1", ttnn.relu_max, t1, 1e30)
        g2 = op("s.sub1", ttnn.subtract, g, t2)
        t3 = op("s.add2", ttnn.add, u, lim)
        t4 = op("s.relu_max2", ttnn.relu_max, t3, 2 * lim)
        u2 = op("s.sub2", ttnn.subtract, t4, lim)
        keep += [t1, t2, t3, t4]
    else:
        g2 = op("s.minimum", ttnn.minimum, g, lim)
        u2 = op("s.clamp", ttnn.clamp, u, min=-lim, max=lim)
    keep += [g2, u2]
    yield
    act = op("s.mul_silu", ttnn.multiply, g2, u2, input_tensor_a_activations=se.silu)
    keep.append(act)
    yield
    out = op(
        "s.linear_down",
        ttnn.linear,
        act,
        se.w2,
        sd=True,
        dtype=ttnn.float32,
        compute_kernel_config=se.ckc,
        core_grid=se.grid,
    )
    keep.append(out)
    return out


def drain(g):
    try:
        while True:
            next(g)
    except StopIteration as e:
        return e.value


def run_overlap(order, rg, sg):  # rg = router stage 1 generator
    """Interleave the two op generators per `order`. Returns (router_result, shared_result)."""
    res = {}

    def step(name, g):
        try:
            next(g)
            return False
        except StopIteration as e:
            res[name] = e.value
            return True

    if order == "rs":
        res["r"] = drain(rg)
        res["s"] = drain(sg)
    elif order == "sr":
        res["s"] = drain(sg)
        res["r"] = drain(rg)
    elif order == "S1R":  # shared gate|up matmul first, then whole router, then the rest of the shared expert
        step("s", sg)
        res["r"] = drain(rg)
        res["s"] = drain(sg)
    elif order == "rr":  # shared first, then alternate one op each
        dr = ds = False
        while not (dr and ds):
            if not ds:
                ds = step("s", sg)
            if not dr:
                dr = step("r", rg)
    else:
        raise ValueError(order)
    return res["r"], res["s"]


def build(md, gate, se, h, mode, k, order, cx, mgr=None):
    """Capture a program of k reps. Returns Prog (outs = outputs of the last rep)."""
    p = Prog(md)
    keep = p.keep
    seg_pre = mode in ("seg2", "seg3", "sd")
    seg_post = mode in ("seg3", "sd")
    if mode == "whole":
        md.load_sub_device_manager(mgr)
    p.begin()
    outs = None
    for _ in range(k):
        hh = (
            h if mode == "whole" else ttnn.multiply(h, 1.0)
        )  # pre op (unconfined eltwise is fatal under a loaded manager)
        keep.append(hh)
        if seg_pre:
            p.cut("load" if mode == "sd" else None, mgr)
        cx.on = mode in ("sd", "whole")
        (score, rank), s = run_overlap(order, router_steps(gate, hh, cx, keep), shared_steps(se, hh, cx, keep))
        if seg_post:
            p.cut("clear" if mode == "sd" else None)
        if mode in ("whole", "single_ns2"):
            r = (rank, rank)  # no stage 2 (topk cannot run under a loaded manager)
        else:
            cx.on = False
            r = router_stage2(gate, score, rank, cx, keep)
        post = s if mode == "whole" else ttnn.add(s, s)  # post op
        keep.append(post)
        outs = (r[0], r[1], s, post)
    p.end()
    if mode == "whole":
        md.clear_loaded_sub_device_manager()
        p.whole = mgr
    cx.on = False
    p.outs = outs
    return p


def warm(md, gate, se, h, order, cx, mgr, with_mgr):
    cx.on = with_mgr
    if with_mgr:
        md.load_sub_device_manager(mgr)
    hh = ttnn.multiply(h, 1.0)
    keep = []
    (score, rank), s = run_overlap(order, router_steps(gate, hh, cx, keep), shared_steps(se, hh, cx, keep))
    if with_mgr:
        md.clear_loaded_sub_device_manager()
    cx.on = False
    router_stage2(gate, score, rank, cx, keep)
    ttnn.add(s, s)
    ttnn.synchronize_device(md)


def wall_ms(md, p, n=20):
    for _ in range(3):
        p.replay()
    ttnn.synchronize_device(md)
    ts = []
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(n):
            p.replay()
        ttnn.synchronize_device(md)
        ts.append((time.perf_counter() - t) / n * 1e3)
    return sorted(ts)[1]


def chain(md, gate, se, h, mode, order, cx, mgr=None):
    out = {}
    for k in (1, 3):
        p = build(md, gate, se, h, mode, k, order, cx, mgr)
        ttnn.synchronize_device(md)
        out[k] = wall_ms(md, p)
        if k == 3:
            nseg = sum(1 for s in p.steps if s[0] == "trace")
        p.release()
    return (out[3] - out[1]) / 2, out[1], nseg


def d0(t, i=0):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[i])


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 300_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_subdevice_overlap(mesh_device):
    md = mesh_device
    text = (
        CONFIG_PATH.read_text()
        .replace("num_shared_experts: 1", "num_shared_experts: 0")
        .replace("  shared_expert_ids_to_devices: fully_replicated\n", "")
    )
    cfg = TTMoEGateConfig.from_yaml(text).model_copy(update={"batch_per_device": T})
    w = load_moe_layer(2, experts=[0])
    gate = DSV41Gate(md, cfg, torch_gate_weight=w["gate_weight"], torch_gate_bias=w["gate_bias"], bias_shift=0.0)
    sid = next(iter(w["shared_w0"]))
    se = DSV41SharedExpert(md, w["shared_w0"][sid], w["shared_w1"][sid], w["shared_w2"][sid])
    rep = ttnn.ReplicateTensorToMesh(md)

    def host_h(seed):
        g = torch.Generator().manual_seed(seed)
        return ttnn.from_torch(
            torch.randn(1, 1, T, D, generator=g).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=rep,
        )

    h = ttnn.from_torch(
        torch.randn(1, 1, T, D).to(torch.bfloat16),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )

    grid = md.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y
    ra = 2  # router sub-device rows
    print(f"SDO grid {gx}x{gy}; sub-device A rows [0,{ra}) router, B rows [{ra},{gy}) shared", flush=True)
    A = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, ra - 1))})
    B = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, ra), ttnn.CoreCoord(gx - 1, gy - 1))})
    mgr = md.create_sub_device_manager([ttnn.SubDevice([A]), ttnn.SubDevice([B])], 0)
    cxA, cxB = Ctx(), Ctx()

    # one context object: router ops on A, shared ops on B -> separate contexts selected by op-name prefix
    class Both(Ctx):
        def op(self, name, fn, *a, sd=False, confine=True, **kw):
            c = cxA if name.startswith("r.") else cxB
            c.on = self.on
            return c.op(name, fn, *a, sd=sd, confine=confine, **kw)

    cxA.cores, cxA.sd = A, ttnn.SubDeviceId(0)
    cxB.cores, cxB.sd = B, ttnn.SubDeviceId(1)
    cx = Both()
    cx0 = Ctx()  # never on -> baseline

    # ---------------- eager probe: which ops can be confined (manager loaded)
    warm(md, gate, se, h, "rs", cx0, mgr, False)
    warm(md, gate, se, h, "rs", cx, mgr, True)
    print("SDO confined router ops:", sorted(cxA.supported), flush=True)
    print("SDO UNCONFINED router ops:", sorted(cxA.unsupported), flush=True)
    print("SDO confined shared ops:", sorted(cxB.supported), flush=True)
    print("SDO UNCONFINED shared ops:", sorted(cxB.unsupported), flush=True)

    # ---------------- (A) baseline and (C) segment-boundary cost
    res = {}
    for mode in ("single", "seg2", "seg3"):
        per, base, nseg = chain(md, gate, se, h, mode, "rs", cx0)
        res[mode] = per
        print(
            f"SDO RESULT mode={mode:7s} order=rs per-rep {per:.4f} ms (1 rep wall {base:.4f}, segs/3reps {nseg})",
            flush=True,
        )
    print(
        f"SDO RESULT boundary cost: seg2-single {(res['seg2']-res['single'])*1e3:.1f} us per boundary; seg3-single {(res['seg3']-res['single'])/2*1e3:.1f} us per boundary",
        flush=True,
    )

    ALT[0] = True
    warm(md, gate, se, h, "rs", cx0, mgr, False)
    warm(md, gate, se, h, "rs", cx, mgr, True)
    print(
        "SDO ALT confined shared ops:",
        sorted(cxB.supported),
        " unconfined:",
        sorted(cxB.unsupported),
        " fatal:",
        sorted(cxB.fatal),
        flush=True,
    )
    for mode in ("single", "seg3"):
        per, base, nseg = chain(md, gate, se, h, mode, "rs", cx0)
        res[mode + "_alt"] = per
        print(f"SDO RESULT ALT-clamp mode={mode:7s} order=rs per-rep {per:.4f} ms", flush=True)
    per, base, nseg = chain(md, gate, se, h, "single_ns2", "rs", cx0)
    res["single_ns2"] = per
    print(f"SDO RESULT mode=single_ns2 (no router stage 2) per-rep {per:.4f} ms", flush=True)

    # ---------------- (B) sub-device overlapped, segmented, several op orders
    for order in ("rs", "sr", "S1R", "rr"):
        warm(md, gate, se, h, order, cx, mgr, True)
        per, base, nseg = chain(md, gate, se, h, "sd", order, cx, mgr)
        print(
            f"SDO RESULT mode=sd      order={order:3s} per-rep {per:.4f} ms (1 rep wall {base:.4f}, segs/3reps {nseg}); vs seg3 no-sd {res['seg3']:.4f}",
            flush=True,
        )

    # ---------------- (D) one manager loaded for the whole trace
    for order in ("rs", "S1R", "rr"):
        try:
            warm(md, gate, se, h, order, cx, mgr, True)
            per, base, nseg = chain(md, gate, se, h, "whole", order, cx, mgr)
            print(
                f"SDO RESULT mode=whole   order={order:3s} per-rep {per:.4f} ms (1 rep wall {base:.4f}, segs {nseg}); vs single_ns2 {res['single_ns2']:.4f}",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"SDO RESULT mode=whole order={order} FAILED {str(e).splitlines()[0][:200]}", flush=True)

    # ---------------- correctness: baseline vs sd (best-effort order S1R) over several replays
    pa = build(md, gate, se, h, "single", 1, "rs", cx0)
    pb = build(md, gate, se, h, "sd", 1, "S1R", cx, mgr)
    pw = None
    try:
        pw = build(md, gate, se, h, "whole", 1, "S1R", cx, mgr)
    except Exception as e:  # noqa: BLE001
        print("SDO whole build failed", str(e).splitlines()[0][:200], flush=True)
    ok = True
    for seed in range(4):
        ttnn.copy_host_to_device_tensor(host_h(seed), h)
        outs = {}
        ref = d0(se.forward(h)).float()
        for name, p in (("A", pa), ("B", pb), ("W", pw)):
            if p is None:
                continue
            p.replay()
            ttnn.synchronize_device(md)
            outs[name] = [[d0(t, i) for i in (0, 17, 31)] for t in p.outs[:3]]
        for name in outs:
            print(
                f"SDO REF seed={seed} {name}: shared pcc vs eager {pcc(outs[name][2][0].float(), ref):.5f} shape {tuple(outs[name][2][0].shape)}",
                flush=True,
            )
        if "W" in outs:
            print(
                f"SDO REF seed={seed} B-vs-W shared bit-equal {torch.equal(outs['B'][2][0], outs['W'][2][0])}",
                flush=True,
            )
        for name in outs:
            if name == "A":
                continue
            for j, label in enumerate(("weights", "indices", "shared")):
                if name == "W" and j < 2:
                    continue
                for di in range(3):
                    a, b = outs["A"][j][di], outs[name][j][di]
                    same = torch.equal(a, b)
                    ok &= same
                    if not same or di == 0:
                        print(
                            f"SDO CORR seed={seed} A-vs-{name} {label} dev{di} bit-equal={same} pcc={pcc(a.float(), b.float()) if not same else 1.0}",
                            flush=True,
                        )
    print("SDO CORRECT_ALL", ok, flush=True)
    pa.release()
    pb.release()
    if pw:
        pw.release()
    md.remove_sub_device_manager(mgr)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 300_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_confinement_probe(mesh_device):
    """Eager probe under a loaded 2-sub-device manager: which router ops accept sub_core_grids / sub_device_id,
    and what happens to an op given neither."""
    md = mesh_device
    grid = md.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y
    rs = lambda x0, y0, x1, y1: ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))})
    A, B = rs(0, 0, gx - 1, 1), rs(0, 2, gx - 1, gy - 1)
    mgr = md.create_sub_device_manager([ttnn.SubDevice([A]), ttnn.SubDevice([B])], 0)
    rep = ttnn.ReplicateTensorToMesh(md)
    mk = lambda *shape, dt=ttnn.float32: ttnn.from_torch(
        torch.rand(*shape),
        device=md,
        dtype=dt,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    E, k = 384, 6
    score = mk(1, 1, T, E)
    bias = mk(1, 1, 1, E)
    idx = ttnn.typecast(ttnn.topk(score, k=k, dim=-1)[1], ttnn.uint16)  # unconfined, manager not loaded yet
    onehot = mk(T, 1, k, E)
    sel = mk(T, 1, k, 1)
    ar = mk(1, 1, 1, E)
    ttnn.synchronize_device(md)
    md.load_sub_device_manager(mgr)

    def probe(name, fn):
        try:
            r = fn()
            ttnn.synchronize_device(md)
            print(f"SDO PROBE {name:42s} OK", flush=True)
            return r
        except Exception as e:  # noqa: BLE001
            print(f"SDO PROBE {name:42s} FAIL {type(e).__name__} {' '.join(str(e).split())[:170]}", flush=True)

    probe("add NO confinement (loaded mgr)", lambda: ttnn.add(score, bias))
    probe("add sub_core_grids=A", lambda: ttnn.add(score, bias, sub_core_grids=A))
    probe("softplus sub_core_grids=A", lambda: ttnn.softplus(score, beta=1.0, threshold=20.0, sub_core_grids=A))
    probe("sqrt sub_core_grids=A", lambda: ttnn.sqrt(score, sub_core_grids=A))
    probe("typecast sub_core_grids=A", lambda: ttnn.typecast(idx, ttnn.float32, sub_core_grids=A))
    probe("eq sub_core_grids=A", lambda: ttnn.eq(onehot, ar, sub_core_grids=A))
    probe("multiply sub_core_grids=A", lambda: ttnn.multiply(onehot, ar, sub_core_grids=A))
    probe("div sub_core_grids=A", lambda: ttnn.div(sel, sel, sub_core_grids=A))
    probe("to_layout RM sub_core_grids=A", lambda: ttnn.to_layout(sel, ttnn.ROW_MAJOR_LAYOUT, sub_core_grids=A))
    probe(
        "matmul sd=A core_grid 2x8",
        lambda: ttnn.matmul(
            mk(1, 1, T, D, dt=ttnn.bfloat16),
            mk(1, 1, D, E),
            dtype=ttnn.float32,
            core_grid=ttnn.CoreGrid(y=2, x=8),
            sub_device_id=ttnn.SubDeviceId(0),
        ),
    )
    probe(
        "matmul sd=B core_grid 8x8",
        lambda: ttnn.matmul(
            mk(1, 1, T, D, dt=ttnn.bfloat16),
            mk(1, 1, D, 4608),
            dtype=ttnn.float32,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            sub_device_id=ttnn.SubDeviceId(1),
        ),
    )
    probe(
        "slice sub_core_grids=B", lambda: ttnn.slice(mk(1, 1, T, 4608), [0, 0, 0, 0], [1, 1, T, 2304], sub_core_grids=B)
    )
    probe("sum(dim=-1) sub_core_grids=A (try)", lambda: ttnn.sum(onehot, dim=-1, keepdim=True, sub_core_grids=A))
    probe("minimum sub_core_grids=B", lambda: ttnn.minimum(mk(1, 1, T, 2304), 10.0, sub_core_grids=B))
    probe("clamp sub_core_grids=B", lambda: ttnn.clamp(mk(1, 1, T, 2304), min=-10.0, max=10.0, sub_core_grids=B))
    g = mk(1, 1, T, 2304)
    lim_t = mk(1, 1, 1, 2304)
    probe("minimum(tensor) sub_core_grids=B", lambda: ttnn.minimum(g, lim_t, sub_core_grids=B))
    probe("maximum(tensor) sub_core_grids=B", lambda: ttnn.maximum(g, lim_t, sub_core_grids=B))
    probe("clip sub_core_grids=B", lambda: ttnn.clip(g, -10.0, 10.0, sub_core_grids=B))
    probe("relu_max/min sub_core_grids=B", lambda: ttnn.relu_max(g, 10.0, sub_core_grids=B))
    probe(
        "multiply silu act sub_core_grids=B",
        lambda: ttnn.multiply(
            g, g, input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)], sub_core_grids=B
        ),
    )
    probe(
        "sum via matmul ones sd=A",
        lambda: ttnn.matmul(
            mk(T, 1, k, E),
            mk(1, 1, E, 32),
            dtype=ttnn.float32,
            core_grid=ttnn.CoreGrid(y=2, x=8),
            sub_device_id=ttnn.SubDeviceId(0),
        ),
    )
    for sname, cores in (
        ("A(2 rows)", A),
        ("row0", rs(0, 0, gx - 1, 0)),
        ("1x12 B row2", rs(0, 2, gx - 1, 2)),
        ("single core", rs(0, 0, 0, 0)),
        ("B", B),
    ):
        probe(
            f"topk sub_core_grids={sname}",
            lambda: ttnn.topk(score, k=k, dim=-1, largest=True, sorted=True, sub_core_grids=cores),
        )
    probe("topk NO confinement (loaded mgr)", lambda: ttnn.topk(score, k=k, dim=-1, largest=True, sorted=True))
    probe("topk NO confinement (loaded mgr)", lambda: ttnn.topk(score, k=k, dim=-1, largest=True, sorted=True))
    probe("sum(dim=-1) NO confinement", lambda: ttnn.sum(onehot, dim=-1, keepdim=True))
    probe("to_layout RM NO confinement", lambda: ttnn.to_layout(sel, ttnn.ROW_MAJOR_LAYOUT))
    md.clear_loaded_sub_device_manager()
    md.remove_sub_device_manager(mgr)
