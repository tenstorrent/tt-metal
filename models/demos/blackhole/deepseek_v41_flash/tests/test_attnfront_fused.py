# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Attention-front POC, steps 1-3 (tt/attn_front.py): wqkv -> q/kv RMSNorm -> wq_b in one program, correctness against torch (all chips compared to chip 0's reference) and trace-replay timing.
Env: AF_STAGE (1 = mm1 only, 2 = + norm, 3 = + qr exchange + mm2; default 3), AF_RING (ring tiles, 240), AF_DSINIT (1), AF_REP (4), AF_TREPS (3), FD_OUT."""

import json
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.attn_front import _shuffle_tiles, attn_front

OUT = os.environ.get("FD_OUT", "/mnt/tt-data/ssinghal/dsv4-logs/attnfront/run_fused")
STAGE = int(os.environ.get("AF_STAGE", "3"))
RING = int(os.environ.get("AF_RING", "240"))
DSINIT = int(os.environ.get("AF_DSINIT", "1"))
REP = int(os.environ.get("AF_REP", "4"))
TREPS = int(os.environ.get("AF_TREPS", "3"))
M, DIM, QL, HD, NQ = int(os.environ.get("AF_T", "8")), 5120, 1280, 512, 4096
EPS = 1e-6
WT_ = ttnn.bfloat8_b if os.environ.get("AF_WDT", "bfp8") == "bfp8" else ttnn.bfloat4_b


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _rms(x, g):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * g


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384, "trace_region_size": 100_000_000}], indirect=True)
@pytest.mark.timeout(2400)
@torch.no_grad()
def test_attnfront_fused(mesh_device):
    md = mesh_device
    os.makedirs(OUT, exist_ok=True)
    rep = ttnn.ReplicateTensorToMesh(md)
    nb = 8
    torch.manual_seed(3)
    x = torch.randn(1, 1, M, DIM)
    if os.environ.get("AF_XDIST") == "t":
        x = torch.distributions.StudentT(2.5).sample((1, 1, M, DIM)) * 0.5
    x = x.bfloat16().float()
    w1 = (torch.randn(1, 1, DIM, QL + HD) * 0.03).bfloat16().float()
    w2 = (torch.randn(1, 1, QL, NQ) * 0.05).bfloat16().float()
    gam = (1.0 + 0.2 * torch.randn(1, 1, 1, QL + HD)).bfloat16().float()

    dram_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(md.dram_grid_size().x - 1, md.dram_grid_size().y - 1))}
    )

    def wt(w, K, N, groups=None):
        return ttnn.from_torch(
            _shuffle_tiles(w, nb, groups),
            dtype=WT_,
            layout=ttnn.TILE_LAYOUT,
            device=md,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(dram_crs, [K, N // nb], ttnn.ShardOrientation.ROW_MAJOR),
            ),
            mesh_mapper=rep,
        )

    def deq(w):  # dequantised weights as the device sees them
        t = ttnn.from_torch(w, dtype=WT_, layout=ttnn.TILE_LAYOUT, device=md, mesh_mapper=rep)
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float().reshape(w.shape[-2], w.shape[-1])

    SPLIT = int(os.environ.get("AF_SPLIT", "1"))
    if SPLIT == 2:
        w1t = wt(torch.nn.functional.pad(w1, (0, 2048 - (QL + HD))), DIM, 2048)
    else:
        w1t = wt(w1, DIM, QL + HD, [2, 2, 2, 1])
    w2t = wt(w2, QL, NQ, [2] * 8)
    xt = None
    if M <= 8:
        xt = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            tile=ttnn.Tile([M, 32]),
            device=md,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
    xmode = int(os.environ.get("AF_XMODE", "1"))
    if xmode >= 2:
        xt = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=md,
            memory_config=ttnn.L1_MEMORY_CONFIG if xmode == 3 else ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )
    xt1 = None
    if xmode == 4:  # diagnosis: x tiny tiles replicated in the L1 of every core (no x delivery in the op)
        from models.demos.blackhole.deepseek_v41_flash.tt.attn_front import core_layout

        _, call = core_layout(md, int(os.environ.get("AF_SPLIT", "1")))
        crs4 = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in call])
        mk4 = lambda t: ttnn.from_torch(
            t.repeat(1, 1, len(call), 1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            tile=ttnn.Tile([8, 32]),
            device=md,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(crs4, [8, DIM], ttnn.ShardOrientation.ROW_MAJOR),
            ),
            mesh_mapper=rep,
        )
        xt = mk4(x[:, :, :8])
        if M > 8:
            xb = torch.zeros(1, 1, 8, DIM)
            xb[:, :, : M - 8] = x[:, :, 8:M]
            xt1 = mk4(xb)
    if xmode == 1:
        cores_ = ttnn.get_optimal_dram_bank_to_logical_worker_assignment(md, ttnn.NOC.RISCV_0_default)
        crs_ = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in cores_])
        xt = ttnn.from_torch(
            x.repeat(1, 1, nb, 1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            tile=ttnn.Tile([M, 32]),
            device=md,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(crs_, [M, DIM], ttnn.ShardOrientation.ROW_MAJOR),
            ),
            mesh_mapper=rep,
        )
    gt = ttnn.from_torch(
        gam.expand(1, 1, 32, -1).contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=md,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rep,
    )

    def z(n):
        return ttnn.from_torch(
            torch.zeros(1, 1, 32, n),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=md,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep,
        )

    qr = z(QL + HD if STAGE == 1 else QL)
    kvn, q = z(HD), z(NQ)

    def run():
        attn_front(
            xt,
            w1t,
            w2t,
            gt,
            qr,
            kvn,
            q,
            stage=STAGE,
            eps=EPS,
            ring_tiles=RING,
            use_ds_init=DSINIT,
            fp32=int(os.environ.get("AF_FP32", "0")),
            x_split=int(os.environ.get("AF_XSPLIT", "0")),
            x_mcast=int(os.environ.get("AF_XMCAST", "0")),
            split=SPLIT,
            x1=xt1,
            T=M,
        )

    run()
    ttnn.synchronize_device(md)
    W1, W2 = deq(w1), deq(w2)
    y = x.reshape(M, DIM) @ W1
    qr_ref = _rms(y[:, :QL], gam[..., :QL].reshape(1, QL))
    kv_ref = _rms(y[:, QL:], gam[..., QL:].reshape(1, HD))
    q_ref = qr_ref.bfloat16().float() @ W2
    res = {}

    def grab(t, n, d):
        return ttnn.to_torch(ttnn.get_device_tensors(t)[d]).float().reshape(32, n)[:M]

    for d in (0, 9, 18, 31):
        if STAGE == 1:
            res[f"y{d}"] = _pcc(grab(qr, QL + HD, d), y)
        else:
            res[f"qr{d}"] = _pcc(grab(qr, QL, d), qr_ref)
            res[f"kv{d}"] = _pcc(grab(kvn, HD, d), kv_ref)
            if STAGE >= 3:
                res[f"q{d}"] = _pcc(grab(q, NQ, d), q_ref)
    print("AFPCC " + json.dumps(res), flush=True)
    if STAGE >= 2 and os.environ.get("AF_DEBUG"):
        got = grab(qr, QL, 0)
        tp = [round(_pcc(got[:, 32 * t : 32 * t + 32], qr_ref[:, 32 * t : 32 * t + 32]), 4) for t in range(QL // 32)]
        rp = [round(_pcc(got[r], qr_ref[r]), 4) for r in range(M)]
        ratio = (got / qr_ref).median(dim=1).values
        print("AFDBG tile pcc", tp, flush=True)
        print("AFDBG row pcc", rp, "median ratio per row", [round(float(v), 4) for v in ratio], flush=True)
        gk = grab(kvn, HD, 0)
        print("AFDBG kv row pcc", [round(_pcc(gk[r], kv_ref[r]), 4) for r in range(M)], flush=True)
        print("AFDBG got[0,:6]", got[0, :6].tolist(), "ref", qr_ref[0, :6].tolist(), flush=True)
    ttnn.ReadDeviceProfiler(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    for _ in range(REP):
        run()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    ttnn.ReadDeviceProfiler(md)
    for _ in range(TREPS):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
        ttnn.ReadDeviceProfiler(md)
    arm = dict(trace_id=int(tid), stage=STAGE, ring=RING, dsinit=DSINIT, pcc=res)
    json.dump(dict(arms={f"stage{STAGE}": arm}, layer=-1), open(os.path.join(OUT, "layer_arms.json"), "w"))
    print("AF_DONE", flush=True)
