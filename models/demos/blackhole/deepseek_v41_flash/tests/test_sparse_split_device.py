# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Key-split sparse_sdpa (tt/sparse_split.py) against ttnn.transformer.sparse_sdpa: PCC / max abs, and eager wall time per call.
Env: SS_SPLITS (default '2,5'), SS_KCHUNK (default 128), SS_ROWS (8), SS_REPS (20)."""

import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import sparse_split as SS

DP = {"l1_small_size": 16384}


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float((a - a.mean()) @ (b - b.mean()) / ((a - a.mean()).norm() * (b - b.mean()).norm() + 1e-30))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param(DP, id="plain")], indirect=True)
@pytest.mark.parametrize("pdt", ["fp8", "bf16"])
@torch.no_grad()
def test_sparse_split(mesh_device, pdt):
    md = mesh_device
    S = int(os.environ.get("SS_ROWS", "8"))
    reps = int(os.environ.get("SS_REPS", "20"))
    kchunk = int(os.environ.get("SS_KCHUNK", "128"))
    splits = [int(x) for x in os.environ.get("SS_SPLITS", "2,5").split(",")]
    R, TOPK, D, H = 20000, 640, 512, 32
    g = torch.Generator().manual_seed(1)
    fp8 = pdt == "fp8"
    kv_t = (
        (torch.randn(1, 1, R, D, generator=g) * 0.5).to(torch.float8_e4m3fn).float()
        if fp8
        else (torch.randn(1, 1, R, D, generator=g) * 0.5)
    )
    q_t = (torch.randn(1, H, S, D, generator=g) * float(os.environ.get("SS_QSCALE", "1"))).to(torch.bfloat16)
    nvs = [640, 640, 600, 513, 300, 129, 128, 1][:S] + [640] * max(0, S - 8)
    idx_t = torch.full((1, 1, S, TOPK), -1, dtype=torch.int32)
    for r in range(S):
        idx_t[0, 0, r, : nvs[r]] = torch.randperm(R, generator=g)[: nvs[r]]
    sink_t = (torch.randn(1, 1, 1, H, generator=g)).to(torch.bfloat16)
    up = lambda t, dt, lay: ttnn.from_torch(
        t,
        device=md,
        dtype=dt,
        layout=lay,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    kv = up(kv_t, ttnn.fp8_e4m3 if fp8 else ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    q = up(q_t, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    idx = up(idx_t, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    sink = up(sink_t, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    ckc = ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    scale = 512**-0.5
    fmt = ttnn.transformer.SparseKVFormat.FP8_E4M3 if fp8 else ttnn.transformer.SparseKVFormat.BF16

    def ref():
        return ttnn.transformer.sparse_sdpa(
            q, kv, idx, D, kv_format=fmt, scale=scale, k_chunk_size=128, compute_kernel_config=ckc, attention_sink=sink
        )

    r0 = ttnn.to_torch(ttnn.get_device_tensors(ref())[0]).float()
    ttnn.synchronize_device(md)
    t0 = time.time()
    for _ in range(reps):
        o = ref()
    ttnn.synchronize_device(md)
    t_ref = (time.time() - t0) / reps * 1e6
    print(f"SS_REF pdt={pdt} S={S}: {t_ref:.1f} us/call (eager wall)")
    for G in splits:
        qin = q
        if os.environ.get("SS_QTILE") == "1":  # real layout: q as [1,S,32,512] TILE, pad rows 9.. = NaN
            qp = q_t.permute(0, 2, 1, 3).clone()
            qp[:, :, 9:, :] = float("nan")
            qin = up(qp, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        f = lambda: SS.sparse_sdpa_split(qin, kv, idx, D, scale, kchunk, G, ckc, sink)
        o = ttnn.to_torch(ttnn.get_device_tensors(f())[0]).float()
        ttnn.synchronize_device(md)
        t0 = time.time()
        for _ in range(reps):
            oo = f()
        ttnn.synchronize_device(md)
        t_s = (time.time() - t0) / reps * 1e6
        worst = min(pcc(o[0, :8, r], r0[0, :8, r]) for r in range(S))
        print(
            f"SS_SPLIT pdt={pdt} G={G} kchunk={kchunk}: {t_s:.1f} us/call, min row PCC (8 real heads) {worst:.6f}, "
            f"max abs {float((o - r0).abs().max()):.5f}, ref max {float(r0.abs().max()):.3f}"
        )
        assert worst > 0.999 or os.environ.get("SS_DBG", "0") != "0"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param(DP, id="plain")], indirect=True)
@pytest.mark.parametrize("topk", [640, 128])
@torch.no_grad()
def test_sparse_split_stress(mesh_device, topk):
    """Repeated calls of the cached program with a fresh random valid count per row every iteration (stale semaphore / CB state, partially valid chunks, topk 128 / 640)."""
    md = mesh_device
    S, R, D, H = 4, 20000, 512, 32
    iters = int(os.environ.get("SS_ITERS", "40"))
    g = torch.Generator().manual_seed(5)
    kv_t = (torch.randn(1, 1, R, D, generator=g) * 0.7).to(torch.float8_e4m3fn).float()
    up = lambda t, dt, lay: ttnn.from_torch(
        t,
        device=md,
        dtype=dt,
        layout=lay,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    kv = up(kv_t, ttnn.fp8_e4m3, ttnn.ROW_MAJOR_LAYOUT)
    sink = up(torch.randn(1, 1, 1, H, generator=g).to(torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    ckc = ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    scale = 512**-0.5
    worst = 1.0
    for it in range(iters):
        q_t = (torch.randn(1, H, S, D, generator=g) * (1 + 6 * (it % 3))).to(torch.bfloat16)
        qp = q_t.permute(0, 2, 1, 3).clone()
        qp[:, :, 9:, :] = float("nan")
        idx_t = torch.full((1, 1, S, topk), -1, dtype=torch.int32)
        for r in range(S):
            nv = int(torch.randint(1, topk + 1, (1,), generator=g))
            idx_t[0, 0, r, :nv] = torch.randperm(R, generator=g)[:nv]
        q = up(q_t, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        qt = up(qp, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        idx = up(idx_t, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        ref = ttnn.to_torch(
            ttnn.get_device_tensors(
                ttnn.transformer.sparse_sdpa(
                    q,
                    kv,
                    idx,
                    D,
                    kv_format=ttnn.transformer.SparseKVFormat.FP8_E4M3,
                    scale=scale,
                    k_chunk_size=128,
                    compute_kernel_config=ckc,
                    attention_sink=sink,
                )
            )[0]
        ).float()
        out = ttnn.to_torch(
            ttnn.get_device_tensors(SS.sparse_sdpa_split(qt, kv, idx, D, scale, 128, 5, ckc, sink))[0]
        ).float()
        p = min(pcc(out[0, 1:9, r], ref[0, 1:9, r]) for r in range(S))
        worst = min(worst, p)
        if p < 0.999:
            print(
                f"SS_STRESS BAD it={it} topk={topk} pcc={p:.5f} nv={[int((idx_t[0,0,r]>=0).sum()) for r in range(S)]}"
            )
    print(f"SS_STRESS topk={topk} worst PCC over {iters} iterations: {worst:.6f}")
    assert worst > 0.999
