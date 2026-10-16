# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Scratch: the Kimi K3 MoE block, token-split (tt/moe/tt_k3_token_split_moe.py), on the LoudBox as a 2 x 4 FABRIC_2D
mesh at the K3 4 x 4 stage's per-chip load: 56 experts per chip (448 here; K3_E to change; the flat expert takes at most
64 per chip, so the real 896 needs the 4 x 4 stage), from a pool of P distinct random experts, top-16, S own tokens per
chip, K3 widths.
Checks the block output against the host model (with the device's routing, and with its own), the routing against
the host routing, and the stages; times one forward per op (device profiler, slowest chip)."""

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

R, C = 2, 4
N = R * C
S = int(os.environ.get("K3_S", "320"))
POOL = int(os.environ.get("K3_POOL", "8"))
ITERS = int(os.environ.get("K3_ITERS", "5"))
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
_PARAMS = [
    pytest.param((R, C), fabric2d_device_params(fabric_payload_size=7232, l1_small_size=0), id="fabric2d-mesh-2x4-2link")
]


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def per_chip_programs(mesh_device):
    """Per chip: [(name, us)] of the programs since the last read."""
    ttnn.ReadDeviceProfiler(mesh_device)
    out = []
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        row = []
        for p in programs:
            res = getattr(p, "program_analyses_results", None) or {}
            if _DURATION_KEY in res:
                row.append((str(getattr(p, "name", "") or getattr(p, "program_name", "") or "?"), float(res[_DURATION_KEY].duration) / 1e3))
        out.append(row)
    return out


@pytest.mark.parametrize("mesh_device, device_params", _PARAMS, indirect=["mesh_device", "device_params"])
@pytest.mark.timeout(7200)
def test_k3_moe_layer(mesh_device, device_params):
    assert tuple(mesh_device.shape) == (R, C)
    d = K3Dims(E=int(os.environ.get("K3_E", str(56 * N))))
    torch.manual_seed(0)
    t0 = time.time()
    w = random_k3_weights(d, pool_size=POOL)
    moe = TtK3TokenSplitMoE(mesh_device, w, d, S)
    logger.info(f"K3 block built in {time.time() - t0:.0f} s: {moe.EPC} experts per chip, combine accumulators {moe.A}, cells {len(moe.cells)}")
    if os.environ.get("K3_WCHECK"):  # (debug: the routed weights read back from the device vs host roundings)
        from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import flat_weight_regions

        lay = dict(moe.op.plan)
        W_l = [(w.pool[e % POOL][0].T.contiguous(), w.pool[e % POOL][1].T.contiguous(), w.pool[e % POOL][2].T.contiguous()) for e in range(moe.EPC)]
        regions, dregs, _ = flat_weight_regions(W_l, lay)
        banks = lay["banks"]
        for name, regs, t in (("gate/up", regions, moe.op.w_gu), ("down", dregs, moe.op.w_d)):
            per = -(-len(regs) // banks)
            regs = list(regs) + [torch.zeros_like(regs[0])] * (per * banks - len(regs))
            host = torch.cat([torch.cat([regs[b + h * banks] for h in range(per)]).reshape(-1, 32) for b in range(banks)], dim=1)
            dev = ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float().reshape(host.shape)
            for lbl, src in (("fp32", host), ("bf16 first", host.to(torch.bfloat16).float())):
                hq = ttnn.to_torch(ttnn.from_torch(src.contiguous(), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()
                logger.info(f"K3 WCHECK {name}: device != host bfp4({lbl}): {(dev != hq).float().mean().item() * 100:.3f}% of elements")
        pytest.skip("weight check")
    if os.environ.get("K3_MEMDUMP"):  # (debug: the L1 map after the build)
        ttnn.device.dump_device_memory_state(mesh_device, prefix=os.environ["K3_MEMDUMP"])
        pytest.skip("memory dump")
    x = torch.randn(N * S, d.H, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0))
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
    logger.info(f"stages: latent down PCC {pcc(lat, xs @ w.lat_down.T):.5f}, shared expert PCC {pcc(ys, a @ w.sh_down.T):.5f}")
    if os.environ.get("K3_DUMP"):  # (the device's outputs per stage, for the offline accuracy analysis)
        cat = lambda t, *shp: torch.cat([ttnn.to_torch(u).reshape(*shp) for u in ttnn.get_device_tensors(t)])  # noqa: E731
        dump = dict(
            S=S, N=N, E=d.E, pool=POOL, wdtype=moe.wdtype, act=moe.act, x=x, gi=gi, gw=gw, logits=cat(inter["logits"], S, d.E), lat=cat(inter["lat"], S, d.latent),
            routed=cat(inter["routed"], S, d.latent), yr=cat(inter["yr"], S, d.H), ys=cat(inter["ys"], S, d.H), y=cat(y, S, d.H),
        )
        torch.save(dump, os.environ["K3_DUMP"])
        logger.info(f"K3 dump: {os.environ['K3_DUMP']}")
    assert pcc(yd, yref_dev) > 0.98, "block output off"
    # timing: per stage (the programs between two stage marks), slowest chip; and the whole forward
    per_chip_programs(mesh_device)
    stage_us, totals, stage_progs = {}, [], {}

    def hook(name):
        ttnn.synchronize_device(mesh_device)
        progs = per_chip_programs(mesh_device)
        stage_us.setdefault(name, []).append((max(sum(u for _, u in row) for row in progs), max(len(r) for r in progs)))
        n = min(len(r) for r in progs)
        stage_progs[name] = [max(r[i][1] for r in progs) for i in range(n)]

    for it in range(ITERS):
        yy = moe(xt, hook=hook)
        ttnn.synchronize_device(mesh_device)
        if it == 0:
            y2 = torch.cat([ttnn.to_torch(t).reshape(S, d.H) for t in ttnn.get_device_tensors(yy)]).float()
            logger.info(f"K3 repeat call: PCC vs the first call {pcc(y2, yd):.6f}, vs host (device routing) {pcc(y2, yref_dev):.5f}, max |diff| {(y2 - yd).abs().max().item():.3e}")
            assert pcc(y2, yref_dev) > 0.98, "repeat call off"
        ttnn.deallocate(yy)
    total = 0.0
    for name, v in stage_us.items():
        med = statistics.median(u for u, _ in v)
        total += med
        logger.info(f"K3_OP {name:16s} {med:9.1f} us  ({v[-1][1]} programs: {' '.join(f'{u:.1f}' for u in stage_progs[name])})")
    logger.info(f"K3_MOE block (sum of the stages' device times, slowest chip per stage, median over {ITERS}): {total:.0f} us")
