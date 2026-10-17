# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Scratch: the Kimi K3 MoE block, token-split (tt/moe/tt_k3_token_split_moe.py), on the LoudBox as a 2 x 4 FABRIC_2D
mesh at the K3 4 x 4 stage's per-chip load: 56 experts per chip (448 here; K3_E to change; the flat expert takes at most
64 per chip, so the real 896 needs the 4 x 4 stage), from a pool of P distinct random experts, top-16, S own tokens per
chip, K3 widths.
Checks the block output against the host model (with the device's routing, and with its own), the routing against
the host routing, and the stages; times one forward per op (device profiler, slowest chip)."""

import math
import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tt.moe.tt_k3_token_split_moe import (
    K3Dims,
    TtK3TokenSplitMoE,
    random_k3_weights,
    reference_k3_moe,
)

# mesh / fabric: K3_MESH=RxC (2x4 LoudBox default; 4x2, 4x4, 8x4 on a Galaxy), K3_FABRIC=2d | torus_x | torus_y |
# torus_xy. S own tokens per chip (default 320; the K3 chunk of 5120 tokens is S = 5120 / N on 4x4 / 8x4)
R, C = (int(v) for v in os.environ.get("K3_MESH", "2x4").split("x"))
N = R * C
# the mesh opened (K3_SYS_MESH, default K3_MESH): on a Galaxy open the whole system mesh (4x8) with its fabric and run
# on a submesh (opening only a subset of the chips leaves their fabric routers' Ethernet peers unstarted: handshake hang)
SR, SC = (int(v) for v in os.environ.get("K3_SYS_MESH", f"{R}x{C}").split("x"))
# K3 on a Galaxy stage: all 896 experts (56 a chip on 4x4, 28 on 8x4) and a 5120-token chunk (S = 5120 / N); on an
# 8-chip mesh 56 experts a chip (the 4x4 stage's load) and S = 320
E_ALL = int(os.environ.get("K3_E", str(896 if N >= 16 else 56 * N)))
S = int(os.environ.get("K3_S", str(5120 // N if N >= 16 else 320)))
_FABRIC = os.environ.get("K3_FABRIC", "2d")
_FCFG = {
    "2d": ttnn.FabricConfig.FABRIC_2D,
    "torus_x": ttnn.FabricConfig.FABRIC_2D_TORUS_X,
    "torus_y": ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
    "torus_xy": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
}[_FABRIC]
# (every chip's local expert e uses pool[e % P]: P must divide the experts per chip - 28 on 8x4 takes 4)
POOL = math.gcd(int(os.environ.get("K3_POOL", "8")), E_ALL // N)
ITERS = int(os.environ.get("K3_ITERS", "5"))
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
_PARAMS = [
    pytest.param(
        (SR, SC),
        fabric2d_device_params(fabric_payload_size=7232, l1_small_size=0, fabric_config=_FCFG),
        id=f"fabric-{_FABRIC}-mesh-{R}x{C}-2link",
    )
]


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def k3_mesh(mesh_device):
    """The mesh the block runs on: the opened one, or its (R, C) submesh."""
    logger.info(f"K3 opened mesh {tuple(mesh_device.shape)}, running on {(R, C)}")
    if tuple(mesh_device.shape) == (R, C):
        return mesh_device
    return mesh_device.create_submesh(ttnn.MeshShape(R, C))


def k3_torus():
    """(wrap X, wrap Y): the fabric's rings that the (sub)mesh spans end to end."""
    wx = _FABRIC in ("torus_x", "torus_xy") and C == SC and C > 2
    wy = _FABRIC in ("torus_y", "torus_xy") and R == SR and R > 2
    return (wx, wy)


def per_chip_programs(mesh_device):
    """Per chip: [(name, us)] of the programs since the last read."""
    ttnn.ReadDeviceProfiler(mesh_device)
    out = []
    data = ttnn.get_latest_programs_perf_data() or {}
    for dev, programs in data.items():
        row = []
        for p in programs:
            res = getattr(p, "program_analyses_results", None) or {}
            if _DURATION_KEY in res:
                row.append(
                    (
                        str(getattr(p, "name", "") or getattr(p, "program_name", "") or "?"),
                        float(res[_DURATION_KEY].duration) / 1e3,
                    )
                )
        out.append(row)
    if not any(out):
        logger.warning(f"K3 profiler: no program data ({len(data)} devices in the read)")
    return out


@pytest.mark.parametrize("mesh_device, device_params", _PARAMS, indirect=["mesh_device", "device_params"])
@pytest.mark.timeout(7200)
def test_k3_moe_layer(mesh_device, device_params):
    mesh_device = k3_mesh(mesh_device)
    d = K3Dims(E=E_ALL)
    torch.manual_seed(0)
    t0 = time.time()
    w = random_k3_weights(d, pool_size=POOL)
    moe = TtK3TokenSplitMoE(mesh_device, w, d, S, torus=k3_torus())
    logger.info(
        f"K3 block built in {time.time() - t0:.0f} s: {moe.EPC} experts per chip, combine accumulators {moe.A}, cells {len(moe.cells)}"
    )
    cl1 = getattr(moe.combine, "l1", None)
    if cl1:
        logger.info(
            f"K3 combine L1: ring depth {moe.combine.depth}, rings {cl1['rings'] >> 10} KB + state {cl1['state'] >> 10} KB "
            f"per accumulator (acc area {cl1['acc_area'] >> 10} KB, port area {cl1['port_area'] >> 10} KB), torus {moe.torus}"
        )
    if os.environ.get("K3_WCHECK"):  # (debug: the routed weights read back from the device vs host roundings)
        from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import flat_weight_regions

        lay = dict(moe.op.plan)
        W_l = [
            (w.pool[e % POOL][0].T.contiguous(), w.pool[e % POOL][1].T.contiguous(), w.pool[e % POOL][2].T.contiguous())
            for e in range(moe.EPC)
        ]
        regions, dregs, _ = flat_weight_regions(W_l, lay)
        banks = lay["banks"]
        for name, regs, t in (("gate/up", regions, moe.op.w_gu), ("down", dregs, moe.op.w_d)):
            per = -(-len(regs) // banks)
            regs = list(regs) + [torch.zeros_like(regs[0])] * (per * banks - len(regs))
            host = torch.cat(
                [torch.cat([regs[b + h * banks] for h in range(per)]).reshape(-1, 32) for b in range(banks)], dim=1
            )
            dev = ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float().reshape(host.shape)
            for lbl, src in (("fp32", host), ("bf16 first", host.to(torch.bfloat16).float())):
                hq = ttnn.to_torch(
                    ttnn.from_torch(src.contiguous(), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
                ).float()
                logger.info(
                    f"K3 WCHECK {name}: device != host bfp4({lbl}): {(dev != hq).float().mean().item() * 100:.3f}% of elements"
                )
        pytest.skip("weight check")
    if os.environ.get("K3_MEMDUMP"):  # (debug: the L1 map after the build)
        ttnn.device.dump_device_memory_state(mesh_device, prefix=os.environ["K3_MEMDUMP"])
        pytest.skip("memory dump")
    x = torch.randn(N * S, d.H, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    xt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    y, inter = moe(xt, keep=True)
    ttnn.synchronize_device(mesh_device)
    yd = torch.cat([ttnn.to_torch(t).reshape(S, d.H) for t in ttnn.get_device_tensors(y)]).float()
    g = ttnn.to_torch(ttnn.get_device_tensors(inter["gi"])[0]).to(torch.int64).reshape(N * S, 2 * d.K) & 0xFFFF
    gi = g[:, : d.K]  # (the joint routing rows: ids, then the weights' bf16 bits)
    gw = g[:, d.K :].to(torch.int16).view(torch.bfloat16).float()
    # the reference with the device's routing (the rest of the block), and with its own (routing agreement)
    yref_dev, _, _ = reference_k3_moe(x, w, d, ids=gi, wts=gw)
    yref, ids_ref, wts_ref = reference_k3_moe(x, w, d)
    agree = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(gi, ids_ref)) / (N * S * d.K)
    logger.info(
        f"K3 MoE block: PCC vs host (device routing) {pcc(yd, yref_dev):.5f}, vs host (host routing) {pcc(yd, yref):.5f}; "
        f"routing: {100 * agree:.2f}% of (token, expert) picks equal, gate weights PCC "
        f"{pcc(torch.sort(gw, 1)[0], torch.sort(wts_ref, 1)[0]):.5f}"
    )
    # stages
    lat = torch.cat([ttnn.to_torch(t).reshape(S, d.latent) for t in ttnn.get_device_tensors(inter["lat"])]).float()
    ys = torch.cat([ttnn.to_torch(t).reshape(S, d.H) for t in ttnn.get_device_tensors(inter["ys"])]).float()
    xs = x.float()
    gs, us = xs @ w.sh_gate.T, xs @ w.sh_up.T
    a = d.beta * torch.tanh(gs / d.beta) * torch.sigmoid(gs) * (d.linear_beta * torch.tanh(us / d.linear_beta))
    logger.info(
        f"stages: latent down PCC {pcc(lat, xs @ w.lat_down.T):.5f}, shared expert PCC {pcc(ys, a @ w.sh_down.T):.5f}"
    )
    if os.environ.get("K3_DEBUG_REPEAT"):  # (debug: a second call with the intermediates, diffed against the first)
        snap = lambda t: [ttnn.to_torch(u) for u in ttnn.get_device_tensors(t)]  # noqa: E731
        first = {k: snap(inter[k]) for k in ("gi", "lat")}
        first.update({f"pl.{k}": snap(v) for k, v in inter["pl"].items()})
        y2_, inter2 = moe(xt, keep=True)
        ttnn.synchronize_device(mesh_device)
        second = {k: snap(inter2[k]) for k in ("gi", "lat")}
        second.update({f"pl.{k}": snap(v) for k, v in inter2["pl"].items()})
        for k in first:
            diffs = [int((a.to(torch.float64) != b.to(torch.float64)).sum()) for a, b in zip(first[k], second[k])]
            logger.info(f"K3 DEBUG_REPEAT {k}: elements differing per chip {diffs}")
        y2t = torch.cat([ttnn.to_torch(t).reshape(S, d.H) for t in ttnn.get_device_tensors(y2_)]).float()
        logger.info(f"K3 DEBUG_REPEAT y: PCC vs first {pcc(y2t, yd):.6f}")
    if os.environ.get("K3_DUMP"):  # (the device's outputs per stage, for the offline accuracy analysis)
        cat = lambda t, *shp: torch.cat(
            [ttnn.to_torch(u).reshape(*shp) for u in ttnn.get_device_tensors(t)]
        )  # noqa: E731
        dump = dict(
            S=S,
            N=N,
            E=d.E,
            pool=POOL,
            wdtype=moe.wdtype,
            act=moe.act,
            x=x,
            gi=gi,
            gw=gw,
            logits=cat(inter["logits"], S, d.E),
            lat=cat(inter["lat"], S, d.latent),
            routed=cat(inter["routed"], S, d.latent),
            yr=cat(inter["yr"], S, d.H),
            ys=cat(inter["ys"], S, d.H),
            y=cat(y, S, d.H),
        )
        torch.save(dump, os.environ["K3_DUMP"])
        logger.info(f"K3 dump: {os.environ['K3_DUMP']}")
    assert pcc(yd, yref_dev) > 0.98 or os.environ.get("K3_PROBE_RUN"), "block output off"
    # timing: per stage (the programs between two stage marks), slowest chip; and the whole forward
    per_chip_programs(mesh_device)
    stage_us, totals, stage_progs = {}, [], {}

    def hook(name):
        ttnn.synchronize_device(mesh_device)
        progs = [r for r in per_chip_programs(mesh_device) if r]
        if not progs:
            return
        stage_us.setdefault(name, []).append((max(sum(u for _, u in row) for row in progs), max(len(r) for r in progs)))
        n = min(len(r) for r in progs)
        stage_progs[name] = [max(r[i][1] for r in progs) for i in range(n)]

    for it in range(ITERS):
        yy = moe(xt, hook=hook)
        ttnn.synchronize_device(mesh_device)
        if it == 0:
            y2 = torch.cat([ttnn.to_torch(t).reshape(S, d.H) for t in ttnn.get_device_tensors(yy)]).float()
            logger.info(
                f"K3 repeat call: PCC vs the first call {pcc(y2, yd):.6f}, vs host (device routing) {pcc(y2, yref_dev):.5f}, max |diff| {(y2 - yd).abs().max().item():.3e}"
            )
            assert pcc(y2, yref_dev) > 0.98 or os.environ.get("K3_PROBE_RUN"), "repeat call off"
        ttnn.deallocate(yy)
    total = 0.0
    for name, v in stage_us.items():
        med = statistics.median(u for u, _ in v)
        total += med
        logger.info(
            f"K3_OP {name:16s} {med:9.1f} us  ({v[-1][1]} programs: {' '.join(f'{u:.1f}' for u in stage_progs[name])})"
        )
    logger.info(
        f"K3_MOE block (sum of the stages' device times, slowest chip per stage, median over {ITERS}): {total:.0f} us"
    )


_TRACE_PARAMS = [
    pytest.param(
        (SR, SC),
        fabric2d_device_params(
            fabric_payload_size=7232,
            l1_small_size=0,
            fabric_config=_FCFG,
            trace_region_size=int(os.environ.get("K3_TRACE_MB", "64")) << 20,
        ),
        id=f"fabric-{_FABRIC}-mesh-{R}x{C}-2link-trace",
    )
]


@pytest.mark.parametrize("mesh_device, device_params", _TRACE_PARAMS, indirect=["mesh_device", "device_params"])
@pytest.mark.timeout(7200)
def test_k3_moe_layer_trace(mesh_device, device_params):
    """The whole forward as one trace: wall time per replay (no host dispatch, no syncs between ops), vs the
    untraced forward (host-dispatched, one sync per call / back to back), and the replay's output vs the untraced."""
    mesh_device = k3_mesh(mesh_device)
    d = K3Dims(E=E_ALL)
    w = random_k3_weights(d, pool_size=POOL)
    moe = TtK3TokenSplitMoE(mesh_device, w, d, S, torus=k3_torus())
    x = torch.randn(N * S, d.H, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    xt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    host = lambda t: torch.cat(
        [ttnn.to_torch(u).reshape(S, d.H) for u in ttnn.get_device_tensors(t)]
    ).float()  # noqa: E731
    n = int(os.environ.get("K3_TRACE_ITERS", "20"))
    y0 = moe(xt)  # (compiles every program)
    ttnn.synchronize_device(mesh_device)
    ref = host(y0)
    ttnn.deallocate(y0)
    t0 = time.perf_counter()
    for _ in range(n):  # untraced, one sync per call
        yy = moe(xt)
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(yy)
    t_sync = (time.perf_counter() - t0) / n
    t0 = time.perf_counter()
    for _ in range(n):  # untraced, back to back
        yy = moe(xt)
        ttnn.deallocate(yy)
    ttnn.synchronize_device(mesh_device)
    t_b2b = (time.perf_counter() - t0) / n
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    y_tr = moe(xt)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)  # (warm)
    t0 = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    t_tr = (time.perf_counter() - t0) / n
    got = host(y_tr)
    logger.info(
        f"K3_TRACE forward wall time per call: traced {t_tr * 1e6:.0f} us | untraced back to back {t_b2b * 1e6:.0f} us "
        f"| untraced with a sync per call {t_sync * 1e6:.0f} us (median of {n}); replay vs untraced PCC "
        f"{pcc(got, ref):.6f}, max |diff| {(got - ref).abs().max().item():.3e}"
    )
    if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":  # per program of one replay: durations and the gaps
        ttnn.ReadDeviceProfiler(mesh_device)
        ttnn.get_latest_programs_perf_data()
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        ttnn.ReadDeviceProfiler(mesh_device)
        spans = []
        for chip, programs in enumerate((ttnn.get_latest_programs_perf_data() or {}).values()):
            res = [
                p.program_analyses_results[_DURATION_KEY]
                for p in programs
                if _DURATION_KEY in (p.program_analyses_results or {})
            ]
            res = sorted(res, key=lambda r: r.start_timestamp)
            if not res:
                continue
            busy = sum(r.duration for r in res) / 1e3
            span = res[-1].end_timestamp - res[0].start_timestamp
            spans.append((chip, len(res), busy, res))
            logger.info(
                f"K3_TRACE chip {chip}: {len(res)} programs, busy {busy:.0f} us, first start .. last end {span} cycles"
            )
        if spans:
            chip, cnt, busy, res = max(spans, key=lambda t: t[2])
            f = 1.35e3  # cycles per us (AICLK 1350 MHz)
            t0_ = res[0].start_timestamp
            for r in res:
                logger.info(
                    f"K3_TRACE_OP chip {chip} start {(r.start_timestamp - t0_) / f:8.1f} us  dur {r.duration / 1e3:8.1f} us"
                )
    ttnn.release_trace(mesh_device, tid)
    assert pcc(got, ref) > 0.999 or os.environ.get("K3_PROBE_RUN"), "the trace replay differs from the untraced forward"
