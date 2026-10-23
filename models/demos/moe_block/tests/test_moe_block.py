# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MoEBlock end to end on the LoudBox 2x4 (MOE_BLOCK_WORKLOG.md): random weights through the MoEWeights contract
(routed experts from a pool: global expert e uses pool[e % P]), one forward checked against a torch reference (with the
device's routing, and the routing agreement with the host's), then per-stage device times (TT_METAL_DEVICE_PROFILER=1
TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1) and the wall time of a forward."""

import dataclasses
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
    MoEBlock,
    MoEModel,
    MoEOptions,
    MoEWeights,
    WeightCache,
    act_host,
    fuse_gate_up,
    split_gate_up,
)

R, C = 2, 4
POOL = 4
ITERS = 5
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
MODELS = {
    "glm53_flash": GLM_53_FLASH,
    # (K3 at the 4x4 stage's per-chip load on the 2x4: 448 experts, 56 a chip - the flat op caps at 64 a chip)
    "k3_448": dataclasses.replace(KIMI_K3, num_experts=448),
}


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def random_weights(m: MoEModel, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s, sc=0.02: torch.randn(*s, generator=g) * sc  # noqa: E731
    H, Hr, I, E = m.emb_dim, m.routed_width, m.routed_expert_hidden_dim, m.num_experts
    pool = [{"gate_up": fuse_gate_up(r(I, Hr), r(I, Hr)), "down": r(Hr, I)} for _ in range(POOL)]
    shared = None
    if m.shared_expert_hidden_dim is not None:
        Is = m.shared_expert_hidden_dim
        shared = {"gate_up": fuse_gate_up(r(Is, H), r(Is, H)), "down": r(H, Is)}
    latent = None
    if m.latent_dim is not None:
        latent = {"down": r(Hr, H), "up": r(H, Hr), "norm": 1.0 + r(Hr, sc=0.1)}
    return MoEWeights(
        router={"weight": r(E, H), "bias": r(E, sc=0.01)},
        routed=[pool[e % POOL] for e in range(E)],
        shared=shared,
        latent=latent,
    )


def reference(x, w: MoEWeights, m: MoEModel, ids=None, wts=None):
    """Host model of the block in fp32: y [T, H], ids [T, K], weights [T, K]."""
    x = x.float()
    rt = m.router
    s = torch.sigmoid(x @ w.router["weight"].float().T)
    if ids is None:
        _, ids = torch.topk(s + w.router["bias"].float(), m.top_k, dim=-1)
        wts = s.gather(1, ids)
        wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * rt.route_scale
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
        fabric2d_device_params(fabric_payload_size=fabric_payload(m.routed_width), l1_small_size=0),
        name,
        chunk,
        id=f"{name}-{chunk}",
    )
    for name, m in MODELS.items()
    for chunk in (5120, 1280)  # (1280: the Galaxy's per-chip load, 640 rows a mesh row)
]


@pytest.mark.parametrize("mesh_device, device_params, model, chunk", _PARAMS, indirect=["mesh_device", "device_params"])
@pytest.mark.timeout(3600)
def test_moe_block(mesh_device, device_params, model, chunk):
    import os

    m = MODELS[model]
    T, H = chunk, m.emb_dim
    w = random_weights(m)
    t0 = time.perf_counter()
    # (MOE_HEAD: the head as one fused program, a program a stage, or where built - test-only)
    opts = MoEOptions(head=os.environ.get("MOE_HEAD", "auto"))
    block = MoEBlock(mesh_device, m, chunk, w, WeightCache(path=None, layer=0), options=opts)
    logger.info(f"MoEBlock {model}: built in {time.perf_counter() - t0:.1f} s")
    g = block.geometry
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


_SE_PARAMS = [
    pytest.param(
        (R, C),
        fabric2d_device_params(fabric_payload_size=fabric_payload(m.routed_width), l1_small_size=0),
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
