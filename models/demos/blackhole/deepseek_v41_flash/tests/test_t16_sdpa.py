# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""T16: single-chip sparse_sdpa microbenchmark of the prefill window (fp8 pool, 32 padded heads, k_chunk 128, idx row = 128 window rows + 512 selected rows).

Inputs: synthetic (random top-k) or REAL selected ids (T16_IDS=<file of tt/trace_dump 'ids' [U,1,C,512] int64, from PROBE_DUMPIDS>), window = consecutive scratch rows (paged read mode).
T16_S rows (512 = 4k window per mesh, 2048 = 16k window); T16_REF=<file>: compare the output with a saved output (bit equality + PCC); T16_SAVE=<file>: save this output.
Timing = mean wall of T16_REPS back-to-back calls (async dispatch, one sync); env DSV41_SDPA_BALANCE=1 (production) / DSV41_SDPA_WINDOW_RUN=1 are read by the op."""
import os
import time

import pytest
import torch

import ttnn

S = int(os.environ.get("T16_S", "512"))
REPS = int(os.environ.get("T16_REPS", "20"))
TOPK, W = 512, 128


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_t16_sdpa(device):
    dev = device
    g = torch.Generator().manual_seed(7)
    ids_f = os.environ.get("T16_IDS")
    if ids_f:
        real = torch.load(ids_f).reshape(-1, 512)[:, :TOPK].long()
        reps = -(-S // real.shape[0])
        real = real.repeat(reps, 1)[:S]
        valid = (real >= 0) & (real != 0xFFFFFFFF)
        R_hist = int(real[valid].max().item()) + 1
    else:
        R_hist = 65536
        real = torch.randint(0, R_hist, (S, TOPK), generator=g)
        valid = torch.ones_like(real, dtype=torch.bool)
    scratch = ((R_hist + 31) // 32) * 32
    R = scratch + W + S + 64
    pool = ttnn.from_torch(
        torch.randn(1, 1, R, 512, generator=g).float(),
        device=dev,
        dtype=ttnn.fp8_e4m3,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    q = ttnn.from_torch(
        torch.randn(1, 32, S, 512, generator=g).to(torch.bfloat16),
        device=dev,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    i = torch.arange(S).view(-1, 1)
    c = torch.arange(W).view(1, -1)
    win = scratch + 1 + i + c  # position p = x + i - 127 + c at scratch row base + 1 + i + c (not-first window)
    top = torch.where(valid, real, torch.full_like(real, 0xFFFFFFFF))
    # sentinels must be a contiguous tail
    order = (~valid).float().argsort(dim=1, stable=True)
    top = torch.gather(top, 1, order)
    idx = torch.cat([win, top], dim=1)
    tidx = ttnn.from_torch(
        idx.to(torch.int64).to(torch.int32).reshape(1, 1, S, W + TOPK),
        device=dev,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    sink = ttnn.from_torch(
        (torch.randn(1, 1, 1, 32, generator=g) * 0.1).to(torch.bfloat16),
        device=dev,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def run():
        return ttnn.transformer.sparse_sdpa(
            q,
            pool,
            tidx,
            512,
            kv_format=ttnn.transformer.SparseKVFormat.FP8_E4M3,
            scale=0.044,
            k_chunk_size=int(os.environ.get("T16_KC", "128")),
            compute_kernel_config=ckc,
            attention_sink=sink,
        )

    o = run()
    ttnn.synchronize_device(dev)
    out = ttnn.to_torch(o).float()
    ttnn.deallocate(o)
    if os.environ.get("T16_PROF") == "1":  # device zone profile of 3 runs (TT_METAL_DEVICE_PROFILER=1)
        ttnn.ReadDeviceProfiler(dev)
        for _ in range(3):
            o = run()
            ttnn.deallocate(o)
        ttnn.synchronize_device(dev)
        ttnn.ReadDeviceProfiler(dev)
        print("T16SDPA profile read", flush=True)
        return
    ts = []
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(REPS):
            o = run()
            ttnn.deallocate(o)
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t) / REPS * 1e6)
    gold_msg = ""
    if os.environ.get("T16_GOLD") == "1":  # float64 golden of the first rows (8 real heads): which variant is closer to the truth?
        kvf = ttnn.to_torch(pool).reshape(R, 512).float().double()
        qf = ttnn.to_torch(q).reshape(32, S, 512)[:8].double()
        sk = ttnn.to_torch(sink).reshape(32)[:8].double()
        errs = []
        for t_ in range(0, min(S, 16)):
            row = idx[t_]
            rows = row[(row != 0xFFFFFFFF)].long()
            Kt = kvf[rows]
            sc = (qf[:, t_] @ Kt.T) * 0.044
            lg = torch.cat([sc, (sk * 0.044).view(-1, 1)], dim=1)
            pr = torch.softmax(lg, dim=1)[:, :-1]
            errs.append(((pr @ Kt) - out[0, :8, t_].double()).norm() / (pr @ Kt).norm())
        gold_msg = f" | rel err vs float64 golden (16 rows): mean {torch.stack(errs).mean().item():.3e} max {torch.stack(errs).max().item():.3e}"
    msg = f"T16SDPA S={S} real={bool(ids_f)} us/call {['%.1f' % x for x in ts]} min {min(ts):.1f}"
    ref = os.environ.get("T16_REF")
    if ref and os.path.exists(ref):
        r = torch.load(ref)[
            :, :8
        ].contiguous()  # the 8 real heads (the padded heads' rows are garbage under the T16 dedup kernel)
        out8 = out[:, :8].contiguous()
        a, b = out8.flatten(), r.flatten()
        pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
        msg += f" | vs ref: equal {bool(torch.equal(out8, r))} pcc {pcc:.7f} maxdiff {(a - b).abs().max().item():.5f}"
    sv = os.environ.get("T16_SAVE")
    if sv:
        torch.save(out, sv)
    print(msg + gold_msg, flush=True)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("S", [1, 7, 100, 121, 513])
@pytest.mark.parametrize("mode", ["full", "mixed", "dups"])
def test_t16_sdpa_edge(device, S, mode):
    """T16 K-row-cache kernel vs the same op at k_chunk 128 on ragged shapes (fewer tokens than cores, cores with 0 / 1 tokens), rows with a sentinel tail (nv < 640, the first windows of a prompt),
    and rows with duplicated ids. Compares the 8 real heads (PCC >= 0.9995); a hang or a TT_FATAL fails the test."""
    dev = device
    g = torch.Generator().manual_seed(11 + S)
    R = 20000
    pool = ttnn.from_torch(
        torch.randn(1, 1, R, 512, generator=g).float(),
        device=dev, dtype=ttnn.fp8_e4m3, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    q = ttnn.from_torch(
        torch.cat([torch.randn(1, 8, S, 512, generator=g), torch.zeros(1, 24, S, 512)], dim=1).to(torch.bfloat16),
        device=dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    i = torch.arange(S).view(-1, 1)
    win = 15000 + i + torch.arange(W).view(1, -1)
    base = torch.randint(0, 12000, (1, TOPK), generator=g)
    top = (base + torch.randint(-300, 300, (S, TOPK), generator=g)).clamp(0, 14000)  # neighbours select overlapping rows
    if mode == "dups":
        top[:, 5:15] = top[:, 0:10].clone()
    idx = torch.cat([win, top], dim=1)
    if mode == "mixed":
        nv = torch.where(torch.rand(S, generator=g) < 0.5, torch.full((S,), W + TOPK), torch.randint(1, W + TOPK, (S,), generator=g))
        col = torch.arange(W + TOPK).view(1, -1)
        idx = torch.where(col < nv.view(-1, 1), idx, torch.full_like(idx, 0xFFFFFFFF))
    tidx = ttnn.from_torch(
        idx.to(torch.int64).to(torch.int32).reshape(1, 1, S, W + TOPK),
        device=dev, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    sink = ttnn.from_torch(
        (torch.randn(1, 1, 1, 32, generator=g) * 0.1).to(torch.bfloat16),
        device=dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    outs = []
    for kc in (128, 640):
        o = ttnn.transformer.sparse_sdpa(
            q, pool, tidx, 512, kv_format=ttnn.transformer.SparseKVFormat.FP8_E4M3, scale=0.044,
            k_chunk_size=kc, compute_kernel_config=ckc, attention_sink=sink,
        )
        ttnn.synchronize_device(dev)
        outs.append(ttnn.to_torch(o).float()[:, :8].contiguous())
        ttnn.deallocate(o)
    a, b = outs[0].flatten(), outs[1].flatten()
    ok = torch.isfinite(b).all().item()
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item() if S * 8 > 1 else 1.0
    print(f"T16EDGE S={S} {mode}: finite {ok} pcc(k128, k640) {pcc:.6f} maxdiff {(a - b).abs().max().item():.5f}", flush=True)
    assert ok and pcc >= 0.9995, (S, mode, pcc)
