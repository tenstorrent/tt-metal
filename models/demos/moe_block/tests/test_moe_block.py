# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MoEBlock end to end on the LoudBox 2x4 (MOE_BLOCK_WORKLOG.md): random weights through the MoEWeights contract
(routed experts from a pool: global expert e uses pool[e % P]), one forward checked against a torch reference (with the
device's routing, and the routing agreement with the host's), then per-stage device times (TT_METAL_DEVICE_PROFILER=1
TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1) and the wall time of a forward."""

import collections
import dataclasses
import os
import statistics
import time

import pytest
import torch
from loguru import logger
from ttnn.bringup.flat_routed_expert_ttnn.closeout_models import fabric_payload

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.moe_block.moe_block import (
    GLM_53_FLASH,
    KIMI_K3,
    Geometry,
    MoEBlock,
    MoELayer,
    MoEModel,
    MoEOptions,
    MoEWeights,
    WeightCache,
    act_host,
    fuse_gate_up,
    split_gate_up,
)

# (test-only: MOE_MESH "1x4" with MOE_FABRIC "torus_x" - one 2 x 2 square of the LoudBox as a TP ring of 4, the
# p150_x4_ring_1x4 descriptor and that square's TT_VISIBLE_DEVICES; default the 2 x 4 lines)
R, C = (int(v) for v in os.environ.get("MOE_MESH", "2x4").split("x"))
_FABRIC = {"torus_x": "FABRIC_2D_TORUS_X", "torus_xy": "FABRIC_2D_TORUS_XY"}.get(  # (torus_xy: the Galaxy 8x4)
    os.environ.get("MOE_FABRIC", "torus_xy" if (R, C) == (8, 4) else "")  # (the Galaxy: both axes rings by default)
)


def _fab(**kw):
    p = fabric2d_device_params(**kw)
    if _FABRIC:
        p["fabric_config"] = getattr(ttnn.FabricConfig, _FABRIC)
    return p


POOL = 4
ITERS = 5
TRACE_REGION = 64 * 1024 * 1024
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
MODELS = {
    "glm53_flash": GLM_53_FLASH,
    # (K3 at the 4x4 stage's per-chip load on the 2x4: 448 experts, 56 a chip - the flat op caps at 64 a chip)
    "k3_448": dataclasses.replace(KIMI_K3, num_experts=448),
    # (a Galaxy K3 chip on a TP ring of 4 - MOE_MESH 1x4: the router at the real E 896 (folded 8:1 onto 112 experts, 28
    # a chip as a Galaxy chip's, ~91 rows an expert at chunk 640))
    "k3_ring4": dataclasses.replace(KIMI_K3, num_experts=112, router=dataclasses.replace(KIMI_K3.router, fold=8)),
    # (the same on the 2x4: 224 experts folded 4:1, 28 a chip, ~91 rows an expert at chunk 1280)
    "k3_g224": dataclasses.replace(KIMI_K3, num_experts=224, router=dataclasses.replace(KIMI_K3.router, fold=4)),
}


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def _fold_bias(b, F):
    """A folded router's bias (RouterConfig.fold): of each group of F candidates (one expert) one eligible - group g's
    candidate (5 g) % F - the rest far below: a token's top-K are distinct experts, uniform over them."""
    if F == 1:
        return b
    b = b.clone().view(-1, F)
    keep = (torch.arange(b.shape[0]) * 5) % F
    mask = torch.full_like(b, -1.0e4)
    mask[torch.arange(b.shape[0]), keep] = 0.0
    return (b + mask).view(-1)


def random_weights(m: MoEModel, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s, sc=0.02: torch.randn(*s, generator=g) * sc  # noqa: E731
    H, Hr, I, E = m.emb_dim, m.routed_width, m.routed_expert_hidden_dim, m.num_experts
    F = m.router.fold
    pool = [{"gate_up": fuse_gate_up(r(I, Hr), r(I, Hr)), "down": r(Hr, I)} for _ in range(POOL)]
    shared = None
    if m.shared_expert_hidden_dim is not None:
        Is = m.shared_expert_hidden_dim
        shared = {"gate_up": fuse_gate_up(r(Is, H), r(Is, H)), "down": r(H, Is)}
    latent = None
    if m.latent_dim is not None:
        latent = {"down": r(Hr, H), "up": r(H, Hr), "norm": 1.0 + r(Hr, sc=0.1)}
    return MoEWeights(
        router={"weight": r(E * F, H), "bias": _fold_bias(r(E * F, sc=0.01), F)},
        routed=[pool[e % POOL] for e in range(E)],
        shared=shared,
        latent=latent,
    )


def reference(x, w: MoEWeights, m: MoEModel, ids=None, wts=None):
    """Host model of the block in fp32: y [T, H], ids [T, K], weights [T, K]."""
    x = x.float()
    rt = m.router
    lg = x @ w.router["weight"].float().T
    s = torch.sqrt(torch.nn.functional.softplus(lg)) if rt.score_func == "sqrtsoftplus" else torch.sigmoid(lg)
    if ids is None:
        _, ids = torch.topk(s + w.router["bias"].float(), m.top_k, dim=-1)
        wts = s.gather(1, ids)
        wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * rt.route_scale
        ids = ids // rt.fold  # (candidates -> experts)
    h = x if m.latent_dim is None else x @ w.latent["down"].float().T
    acc = torch.zeros(x.shape[0], m.routed_width)
    for p in range(POOL):  # (every expert of a pool slot at once)
        tok, kk = ((ids % POOL) == p).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        gate, up = split_gate_up(w.routed[p]["gate_up"].float())
        a = act_host(m.routed_expert_activation, h[tok] @ gate.T, h[tok] @ up.T)
        acc.index_add_(0, tok, (a @ w.routed[p]["down"].float().T) * wts[tok, kk].float().unsqueeze(1))
    y = acc
    if m.latent_dim is not None:
        if m.latent_norm:
            y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + m.norm_eps) * w.latent["norm"].float()
        y = y @ w.latent["up"].float().T
    if w.shared is not None:
        gs, us = split_gate_up(w.shared["gate_up"].float())
        y = y + act_host(m.shared_expert_activation, x @ gs.T, x @ us.T) @ w.shared["down"].float().T
    return y, ids, wts


def device_routing(rows, S, K):
    """The joint routing rows (per chip [S / 32, 32 x 2K] uint16, row-major chip order) -> ids, weights [N S, K]."""
    ids, wts = [], []
    for t in rows:
        u = t.to(torch.int64).reshape(S, 2 * K) & 0xFFFF
        ids.append(u[:, :K])
        wb = u[:, K:]
        wts.append(torch.where(wb >= 32768, wb - 65536, wb).to(torch.int16).view(torch.bfloat16).float())
    return torch.cat(ids), torch.cat(wts)


def per_chip_programs(mesh_device):
    """Per chip: [us] of the programs since the last read."""
    ttnn.ReadDeviceProfiler(mesh_device)
    out = []
    data = ttnn.get_latest_programs_perf_data() or {}
    if not getattr(per_chip_programs, "logged", False):
        per_chip_programs.logged = True
        for dev, programs in list(data.items())[:1]:
            p0 = programs[0] if programs else None
            logger.info(
                f"profiler read: {len(data)} devices, {len(programs)} programs on {dev}; first: "
                f"{dict(getattr(p0, 'program_analyses_results', {}) or {}).keys() if p0 is not None else None}"
            )
    for programs in data.values():
        out.append(
            [
                float(p.program_analyses_results[_DURATION_KEY].duration) / 1e3
                for p in programs
                if _DURATION_KEY in (getattr(p, "program_analyses_results", None) or {})
            ]
        )
    return out


_PARAMS = [
    pytest.param(
        (R, C),
        _fab(fabric_payload_size=fabric_payload(m.routed_width), l1_small_size=0),
        name,
        chunk,
        id=f"{name}-{chunk}",
    )
    for name, m in MODELS.items()
    for chunk in (5120, 2048, 1280)  # (1280 / 2048: a Galaxy chip at chunk 5120 / 8192 - 640 / 1024 rows a mesh row)
]


@pytest.mark.parametrize("mesh_device, device_params, model, chunk", _PARAMS, indirect=["mesh_device", "device_params"])
@pytest.mark.timeout(3600)
def test_moe_block(mesh_device, device_params, model, chunk):
    import os

    m = MODELS[model]
    if os.environ.get("MOE_EXPERTS"):  # (test-only: e.g. the Galaxy's experts per chip on the 2x4)
        m = dataclasses.replace(m, num_experts=int(os.environ["MOE_EXPERTS"]))
    T, H = chunk, m.emb_dim
    w = random_weights(m)
    t0 = time.perf_counter()
    # (MOE_HEAD / MOE_TAIL: the head / tail as one fused program, a program a stage, or where built - test-only)
    opts = MoEOptions(
        head=os.environ.get("MOE_HEAD", "auto"),
        tail=os.environ.get("MOE_TAIL", "auto"),
        experts_during_dispatch=bool(int(os.environ.get("MOE_EDD", "0"))),
        residual=os.environ.get("MOE_RESIDUAL", "sp_tp"),  # (owners: each chip's own tokens at full width)
    )
    block = MoEBlock(mesh_device, m, chunk, w, WeightCache(path=None, layer=0), options=opts)
    logger.info(f"MoEBlock {model}: built in {time.perf_counter() - t0:.1f} s")
    g = block.geometry
    if opts.residual == "owners":  # (chip d = r C + c holds tokens [d T/N, (d + 1) T/N) at full width)
        mapper = ttnn.ShardTensorToMesh(mesh_device, dim=2)
        composer = ttnn.ConcatMeshToTensor(mesh_device, dim=2)
    else:
        mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(2, 3))
        composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(R, C), dims=(2, 3))
    # (every call its own x - and so its own routing: stale data from the previous call cannot pass the checks)
    NX = 3 + 2 * ITERS
    xs = [torch.randn(T, H, generator=torch.Generator().manual_seed(1000 + k)).to(torch.bfloat16) for k in range(NX)]
    xts = [
        ttnn.from_torch(
            x_.view(1, 1, T, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mapper
        )
        for x_ in xs
    ]
    nxt = iter(range(NX))

    def run(keep=False):
        block.keep_taps = keep
        k = next(nxt)
        out = block(xts[k])
        ttnn.synchronize_device(mesh_device)
        return out, k

    # correctness: three calls, each on its own x against its own reference (the exchange is built in the first one)
    for call in range(3):
        out, k = run(keep=True)
        x = xs[k]
        y = ttnn.to_torch(out, mesh_composer=composer).reshape(T, H).float()
        ttnn.deallocate(out)
        ids_d, wts_d = device_routing(block.taps()["rows"], g.tokens_per_chip, m.top_k)
        y_dev, _, _ = reference(x, w, m, ids=ids_d, wts=wts_d)
        y_ref, ids_r, wts_r = reference(x, w, m)
        agree = (torch.sort(ids_d, 1)[0] == torch.sort(ids_r, 1)[0]).float().mean().item()
        p_dev, p_ref = pcc(y, y_dev), pcc(y, y_ref)
        logger.info(
            f"MoEBlock {model} call {call}: PCC vs host (device routing) {p_dev:.5f}, vs host (host routing) {p_ref:.5f}; "
            f"routing: {100 * agree:.2f}% of picks equal, gate weights PCC "
            f"{pcc(torch.sort(wts_d, 1)[0], torch.sort(wts_r, 1)[0]):.5f}; exchange builds {block.exchange.builds}"
        )
        assert p_dev > 0.98, f"block output off (PCC {p_dev:.5f})"
    # soak (MOE_SOAK=N, test-only): N more calls over a pool of fresh inputs, every MOE_SOAK_CHECK-th (25) and the
    # last checked against its own reference
    soak, every = int(os.environ.get("MOE_SOAK", "0")), int(os.environ.get("MOE_SOAK_CHECK", "25"))
    pool = list(range(NX))
    for i in range(soak):
        k = pool[i % NX]
        block.keep_taps = i % every == every - 1 or i == soak - 1
        out = block(xts[k])
        if block.keep_taps:
            ttnn.synchronize_device(mesh_device)
            y = ttnn.to_torch(out, mesh_composer=composer).reshape(T, H).float()
            ids_d, wts_d = device_routing(block.taps()["rows"], g.tokens_per_chip, m.top_k)
            p_dev = pcc(y, reference(xs[k], w, m, ids=ids_d, wts=wts_d)[0])
            logger.info(f"MoEBlock {model} soak call {i + 1}: PCC vs host (device routing) {p_dev:.5f}")
            assert p_dev > 0.98, f"soak call {i + 1}: block output off (PCC {p_dev:.5f})"
        elif i % 50 == 49:
            ttnn.synchronize_device(mesh_device)
            logger.info(f"MoEBlock {model} soak progress: {i + 1}")
        ttnn.deallocate(out)
    # timing: per stage (the programs between two stage marks), slowest chip
    if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
        per_chip_programs(mesh_device)
        stage = {}

        def hook(name):
            ttnn.synchronize_device(mesh_device)
            progs = [r_ for r_ in per_chip_programs(mesh_device) if r_]
            if progs:
                stage.setdefault(name, []).append((max(sum(r_) for r_ in progs), max(len(r_) for r_ in progs)))

        block.stage_hook = hook
        for _ in range(ITERS):
            ttnn.deallocate(run()[0])
        block.stage_hook = None
        total = 0.0
        for name, v in stage.items():
            med = statistics.median(u for u, _ in v)
            total += med
            logger.info(f"MOE_STAGE {model} {name:16s} {med:9.1f} us  ({v[-1][1]} programs)")
        logger.info(f"MOE_STAGE {model} sum of the stages (slowest chip each, median of {ITERS}): {total:.0f} us")
    # wall time of a forward (host dispatch included; no stage syncs)
    ws = []
    for _ in range(ITERS):
        ttnn.synchronize_device(mesh_device)
        xt = xts[next(nxt)]
        t0 = time.perf_counter()
        out = block(xt)
        ttnn.synchronize_device(mesh_device)
        ws.append((time.perf_counter() - t0) * 1e6)
        ttnn.deallocate(out)
    logger.info(
        f"MOE_WALL {model}: forward {statistics.median(ws):.0f} us wall (median of {ITERS}; {[round(v) for v in ws]})"
    )


_TRACE_PARAMS = [
    pytest.param(
        (R, C),
        _fab(fabric_payload_size=fabric_payload(m.routed_width), l1_small_size=0, trace_region_size=TRACE_REGION),
        name,
        chunk,
        id=f"{name}-{chunk}",
    )
    for name, m in MODELS.items()
    # (1x4: a Galaxy chip's 640 rows a mesh row; 5120: GLM-5.3-Flash's LoudBox chunk)
    for chunk in ((640,) if R == 1 else (1280, 2048, 5120))
] + [
    pytest.param(  # (the Galaxy: 8 x 4, both axes rings; chunk 5120 = 640 rows a mesh row; MOE_EXPERTS=896 for K3's)
        (8, 4),
        fabric2d_device_params(
            fabric_payload_size=fabric_payload(m.routed_width),
            l1_small_size=0,
            trace_region_size=TRACE_REGION,
            fabric_config=ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
        ),
        name,
        5120,
        id=f"{name}-5120-8x4",
    )
    for name, m in MODELS.items()
]


@pytest.mark.parametrize(
    "mesh_device, device_params, model, chunk", _TRACE_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("timing", ["wall", "device"])  # (device: the profiler's per-program times of the replays)
@pytest.mark.timeout(3600)
def test_moe_block_traced(mesh_device, device_params, model, chunk, timing):
    """One forward (chunk 1280 or 2048) captured as a trace on the block's input buffer and replayed: each replay's x (from a
    pool of fresh inputs, never the previous one) written into the input buffer first, each replay's output checked
    against its own reference (the device's routing, read from the kept routing rows); MOE_TRACE_REPLAYS (test-only)
    replays. Then the wall time of a replay, with and without the input write, vs an eager forward."""
    import os

    if timing == "device" and os.environ.get("TT_METAL_DEVICE_PROFILER") != "1":
        pytest.skip("device timing: the device profiler (TT_METAL_DEVICE_PROFILER=1 at process start)")
    m = MODELS[model]
    if os.environ.get("MOE_EXPERTS"):  # (test-only: e.g. the Galaxy's experts per chip on the 2x4)
        m = dataclasses.replace(m, num_experts=int(os.environ["MOE_EXPERTS"]))
    T, H = chunk, m.emb_dim
    w = random_weights(m)
    opts = MoEOptions(
        head=os.environ.get("MOE_HEAD", "auto"),
        tail=os.environ.get("MOE_TAIL", "auto"),
        router=os.environ.get("MOE_ROUTER", "auto"),  # (test-only: the fused head's router placement)
        experts_drisc=os.environ.get("MOE_EXPERTS_DRISC", "1") != "0",  # (test-only: A/B)
        experts_during_dispatch=bool(int(os.environ.get("MOE_EDD", "0"))),
        residual=os.environ.get("MOE_RESIDUAL", "sp_tp"),  # (owners: each chip's own tokens at full width)
        # (test-only: MOE_COMBINE_SNAP=-14 -> the combine's terms snapped to multiples of 2^-14)
        combine_snap=2.0 ** int(os.environ["MOE_COMBINE_SNAP"]) if os.environ.get("MOE_COMBINE_SNAP") else None,
    )
    block = MoEBlock(mesh_device, m, chunk, w, WeightCache(path=None, layer=0), options=opts)
    g = block.geometry
    if opts.residual == "owners":
        mapper = ttnn.ShardTensorToMesh(mesh_device, dim=2)
        composer = ttnn.ConcatMeshToTensor(mesh_device, dim=2)
    else:
        mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 3))
        composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 3))
    NX = 16
    xs = [torch.randn(T, H, generator=torch.Generator().manual_seed(2000 + k)).to(torch.bfloat16) for k in range(NX)]
    hx = [
        ttnn.from_torch(x_.view(1, 1, T, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
        for x_ in xs
    ]
    x_in = block.input_buffer()
    # k -> (routing rows, reference output); a replay whose routing differs from its x's first (a near tie: the router
    # fin sums in landing order only with MoEOptions(deterministic=False), A/B) is checked against its own routing's
    refs, flips = {}, []
    firsts, bits = {}, []  # (x k -> its first output; per replay of an x seen before with the same routing: elements
    # that differ bitwise from that first output - the sums' order dependence)

    def check(out, k, tag):
        y = ttnn.to_torch(out, mesh_composer=composer).reshape(T, H).float()
        rows = block.taps()["rows"]
        if k not in refs or not all(torch.equal(a, b) for a, b in zip(rows, refs[k][0])):
            ids_d, wts_d = device_routing(rows, g.tokens_per_chip, m.top_k)
            ref = (rows, reference(xs[k], w, m, ids=ids_d, wts=wts_d)[0])
            if k in refs:
                ids = lambda t: (t.to(torch.int64).reshape(-1, 2 * m.top_k) & 0xFFFF)[:, : m.top_k]  # noqa: E731
                n = sum(int((ids(a) != ids(b)).any(-1).sum()) for a, b in zip(rows, refs[k][0]))
                flips.append(n)
                logger.info(f"{tag}: x {k}'s routing differs from its first ({n} tokens' picks: near ties)")
                assert n <= 4, f"{tag}: x {k}'s picks differ in {n} tokens - more than near ties"
            else:
                refs[k] = ref
        else:
            ref = refs[k]
        p_dev = pcc(y, ref[1])
        assert p_dev > 0.98, f"{tag} (x {k}): block output off (PCC {p_dev:.5f})"
        rk = (k, tuple(hash(a.numpy().tobytes()) for a in rows))
        if rk in firsts:
            bits.append(int((y != firsts[rk]).sum()))
        else:
            firsts[rk] = y
        return p_dev

    # eager first (every program built and loaded, the DRISC senders started), on the input buffer as traced
    block.keep_taps = "device"
    for k in range(2):
        ttnn.copy_host_to_device_tensor(hx[k], x_in)
        out = block(x_in)
        ttnn.synchronize_device(mesh_device)
        logger.info(f"MoEBlock {model} eager call {k}: PCC {check(out, k, 'eager'):.5f}")
        ttnn.deallocate(out)
    # (the eager forward's wall, before the capture, taps off: kept rows move the head's DRAM outputs - a rebuild a call)
    block.keep_taps, ew = False, []
    for i in range(ITERS + 1):
        ttnn.copy_host_to_device_tensor(hx[i], x_in)
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        ttnn.deallocate(block(x_in))
        ttnn.synchronize_device(mesh_device)
        ew.append((time.perf_counter() - t0) * 1e6)
    ew, block.keep_taps = ew[1:], "device"
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = block(x_in)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    n = int(os.environ.get("MOE_TRACE_REPLAYS", "48"))
    order = [(3 * i + 2) % NX for i in range(n)]  # (consecutive replays on different inputs)
    ps = []
    for i, k in enumerate(order):
        ttnn.copy_host_to_device_tensor(hx[k], x_in)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        ps.append(check(out, k, f"replay {i}"))
        if i % 50 == 49 or i == n - 1:
            logger.info(
                f"MoEBlock {model} traced: {i + 1} replays checked, PCC min {min(ps):.5f}; routing differed from the "
                f"x's first in {len(flips)} ({sum(flips)} rows)"
            )
    if bits:
        logger.info(
            f"MOE_BITS {model}: {len(bits)} replays of an x with the same routing: {sum(b > 0 for b in bits)} differ "
            f"bitwise from that x's first, {sum(bits)} elements in all (max {max(bits)} a replay, of {T * H})"
        )
    # timing
    REP = 20
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(REP):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    back = (time.perf_counter() - t0) * 1e6 / REP
    one, inw = [], []
    for i in range(ITERS):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        one.append((time.perf_counter() - t0) * 1e6)
        t0 = time.perf_counter()
        ttnn.copy_host_to_device_tensor(hx[i], x_in)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        inw.append((time.perf_counter() - t0) * 1e6)
    if timing == "device":
        replay_device_times(mesh_device, tid, model)
    ttnn.release_trace(mesh_device, tid)
    ttnn.deallocate(out)
    med = statistics.median
    logger.info(
        f"MOE_TRACE {model}: replay {med(one):.0f} us wall (median of {ITERS}), back to back {back:.0f} us a replay "
        f"({REP}), with the input write {med(inw):.0f} us; eager forward {med(ew):.0f} us"
    )


def replay_device_times(mesh_device, tid, model, batches=(10, 40, 100)):
    """The profiler on the trace's replays: per replay (trace_id_counter) and chip its programs' device durations, their
    sum and the replay's span (the first program's start to the last's end: the gaps between programs inside the
    trace), after `batches` replays a read (how many fit the profiler's buffers)."""
    per_chip_programs(mesh_device)  # (flush what the capture / correctness replays left)
    freq = 1.35  # (GHz: the timestamps are device cycles)
    for n in batches:
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(n):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        wall = (time.perf_counter() - t0) * 1e6 / n
        ttnn.ReadDeviceProfiler(mesh_device)
        data = ttnn.get_latest_programs_perf_data() or {}
        reps, other = {}, 0  # (chip, replay) -> [(start, end, duration ns)]
        for chip, progs in data.items():
            for p in progs:
                res = (getattr(p, "program_analyses_results", None) or {}).get(_DURATION_KEY)
                if res is None:
                    continue
                uid = p.program_execution_uid
                if int(uid.trace_id) != int(tid):
                    other += 1
                    continue
                reps.setdefault((chip, int(uid.trace_id_counter)), []).append(
                    (int(res.start_timestamp), int(res.end_timestamp), float(res.duration))
                )
        counts = sorted({len(v) for v in reps.values()})
        per_rep = collections.defaultdict(list)
        for (chip, r), v in reps.items():
            per_rep[r].append((sum(d for _, _, d in v) / 1e3, (max(e for _, e, _ in v) - min(s_ for s_, _, _ in v))))
        sums = [max(x for x, _ in v) for v in per_rep.values()]
        spans = [max(y for _, y in v) / freq / 1e3 for v in per_rep.values()]
        # (per program of a replay, in start order: its duration on the slowest chip, median over the replays)
        byidx = collections.defaultdict(list)
        for r in per_rep:
            chips = [sorted(v) for (c, rr), v in reps.items() if rr == r]
            for i in range(min(len(v) for v in chips)):
                byidx[i].append(max(v[i][2] for v in chips) / 1e3)
        progs = " | ".join(f"{statistics.median(v):.1f}" for _, v in sorted(byidx.items()))
        logger.info(f"MOE_TRACE_DEV {model}: programs (us, in order): {progs}")
        logger.info(
            f"MOE_TRACE_DEV {model}: {n} replays a read -> {len(per_rep)} replays seen ({other} programs not of the "
            f"trace), programs a replay a chip {counts}; slowest chip a replay: programs' device sum "
            f"{statistics.median(sums) if sums else 0:.0f} us, span {statistics.median(spans) if spans else 0:.0f} us "
            f"(median, max {max(spans) if spans else 0:.0f}); replay wall {wall:.0f} us"
        )


_LAYER_PARAMS = [
    pytest.param(
        (R, C),
        _fab(fabric_payload_size=fabric_payload(GLM_53_FLASH.routed_width), l1_small_size=0),
        "glm53_flash",
        chunk,
        id=f"glm53_flash-{chunk}",
    )
    for chunk in (1280, 5120)
]


@pytest.mark.parametrize(
    "mesh_device, device_params, model, chunk", _LAYER_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.timeout(3600)
def test_moe_block_layers(mesh_device, device_params, model, chunk):
    """Two layers (their own random weights) on one block, called in turn: each output checked against its own layer's
    reference and not the other's - the block's layer switch (MoEBlock.forward(x, layer=...): the fused head's weights
    patched into the same programs, the ops head's swapped)."""
    import os

    m = MODELS[model]
    T, H = chunk, m.emb_dim
    opts = MoEOptions(head=os.environ.get("MOE_HEAD", "auto"), residual=os.environ.get("MOE_RESIDUAL", "sp_tp"))
    g = Geometry.of(mesh_device, m, chunk, 2, zigzag=False)
    ws = [random_weights(m, seed=s) for s in (0, 1)]
    layers = [MoELayer.build(mesh_device, g, m, w) for w in ws]
    block = MoEBlock(mesh_device, m, chunk, layers[0], WeightCache(path=None, layer=0), options=opts)
    logger.info(f"MoEBlock {model} layers: {block.summary}")
    if opts.residual == "owners":
        mapper, composer = ttnn.ShardTensorToMesh(mesh_device, dim=2), ttnn.ConcatMeshToTensor(mesh_device, dim=2)
    else:
        mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(2, 3))
        composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(R, C), dims=(2, 3))
    xs = [torch.randn(T, H, generator=torch.Generator().manual_seed(3000 + k)).to(torch.bfloat16) for k in range(3)]
    xts = [
        ttnn.from_torch(
            x_.view(1, 1, T, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mapper
        )
        for x_ in xs
    ]
    for call in range(8):
        li, k = call % 2, call % 3
        block.keep_taps = True
        out = block(xts[k], layer=layers[li])
        ttnn.synchronize_device(mesh_device)
        y = ttnn.to_torch(out, mesh_composer=composer).reshape(T, H).float()
        ttnn.deallocate(out)
        ids_d, wts_d = device_routing(block.taps()["rows"], g.tokens_per_chip, m.top_k)
        p_own = pcc(y, reference(xs[k], ws[li], m, ids=ids_d, wts=wts_d)[0])
        p_other = pcc(y, reference(xs[k], ws[1 - li], m, ids=ids_d, wts=wts_d)[0])
        logger.info(
            f"MoEBlock {model} call {call} (layer {li}, x {k}): PCC own layer {p_own:.5f}, other layer {p_other:.5f}"
        )
        assert p_own > 0.98, f"call {call}: layer {li}'s output off (PCC {p_own:.5f})"
        assert p_other < 0.9, f"call {call}: the output matches the other layer's weights (PCC {p_other:.5f})"


_SE_PARAMS = [
    pytest.param(
        (R, C),
        _fab(fabric_payload_size=fabric_payload(m.routed_width), l1_small_size=0),
        name,
        id=f"{name}-se",
    )
    for name, m in MODELS.items()
]


@pytest.mark.parametrize("mesh_device, device_params, model", _SE_PARAMS, indirect=["mesh_device", "device_params"])
@pytest.mark.timeout(3600)
def test_shared_expert(mesh_device, device_params, model):
    """The shared expert alone (TP-split, chunk 5120): device time and accuracy per math fidelity, vs fp32 on the host
    with the same bf8-quantized-on-device weights' source (fp32 weights): PCC and relative L2 of the summed partials."""
    import os

    from models.demos.moe_block.moe_block import Geometry, MoEDtypes, SharedExpert

    m = MODELS[model]
    T, H = 5120, m.emb_dim
    w = random_weights(m)
    g = Geometry.of(mesh_device, m, T, 2, zigzag=False)
    x = torch.randn(T, H, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    xt = ttnn.from_torch(
        x.view(1, 1, T, H),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(2, 3)),
    )
    x_full = ttnn.all_gather(xt, dim=3, cluster_axis=1)
    gs, us = split_gate_up(w.shared["gate_up"].float())
    ref = act_host(m.shared_expert_activation, x.float() @ gs.T, x.float() @ us.T) @ w.shared["down"].float().T
    for fid in (ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi3):
        se = SharedExpert(mesh_device, g, m, w.shared, dataclasses.replace(MoEDtypes(), shared_fidelity=fid))
        out = se(x_full)
        ttnn.synchronize_device(mesh_device)
        parts = [ttnn.to_torch(t).reshape(T // R, H).float() for t in ttnn.get_device_tensors(out)]
        ttnn.deallocate(out)
        y = torch.cat([sum(parts[r * C : (r + 1) * C]) for r in range(R)])  # (row r's chips hold its tokens' partials)
        rel = ((y - ref).norm() / ref.norm()).item()
        us_ = None
        if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
            per_chip_programs(mesh_device)
            ts = []
            for _ in range(ITERS):
                o = se(x_full)
                ttnn.synchronize_device(mesh_device)
                ts.append(max(sum(r_) for r_ in per_chip_programs(mesh_device) if r_))
                ttnn.deallocate(o)
            us_ = statistics.median(ts)
        logger.info(
            f"MOE_SE {model} {fid}: PCC {pcc(y, ref):.5f}, rel L2 {100 * rel:.2f}%, "
            f"{'-' if us_ is None else round(us_, 1)} us"
        )
