# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Lever L12 microbench: per-stage device time of the spec-verify indexer (SpecIndexer: score / mask / topk) on ONE chip of the 4x8 mesh,
production shapes (U = 4 users, n = 1 + k rows, matmul backend, bfp8 key slab). Each stage is timed as a captured trace replayed REPS times.
Env: IDXB_N (entries of the scored slab, comma list), IDXB_ROWS (n rows per user, default 4), IDXB_REPS."""
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import indexer as I
from models.demos.blackhole.deepseek_v41_flash.tt.spec_paged import SpecIndexer

NS = [int(x) for x in os.environ.get("IDXB_N", "32832,65568,131104").split(",")]
NROW = int(os.environ.get("IDXB_ROWS", "4"))
REPS = int(os.environ.get("IDXB_REPS", "20"))


def traced_us(md, fn, reps=REPS):
    fn()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    fn()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    t = time.perf_counter()
    for _ in range(reps):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    us = (time.perf_counter() - t) / reps * 1e6
    ttnn.release_trace(md, tid)
    return us


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384, "trace_region_size": 200_000_000}], indirect=True)
@torch.no_grad()
def test_idx_bench(mesh_device):
    md = mesh_device
    U, n = int(os.environ.get("IDXB_U", "1")), NROW
    R = U * n
    torch.manual_seed(0)
    rep = ttnn.ReplicateTensorToMesh(md)
    for N in NS:
        w = {"wq_b": torch.randn(4096, 1280) * 0.02, "weights_proj": torch.randn(32, 5120) * 0.02}
        freqs = torch.polar(torch.ones(N * 2 + 64, 32), torch.rand(N * 2 + 64, 32))
        idx = SpecIndexer(
            md,
            w,
            freqs,
            users_per_row=U,
            n_alloc=N,
            ratio=2,
            key_dtype=ttnn.bfloat8_b,
            fp4_q=True,
            backend="matmul",
            n=n,
        )
        keys = torch.randn(U, N - 32, 128)
        idx.load_keys(keys)
        idx.set_pos_bucket(None)
        up = lambda t, dt, lay: ttnn.from_torch(t.contiguous(), device=md, dtype=dt, layout=lay, mesh_mapper=rep)
        q = up(torch.randn(U, 1, 32 * n, 128), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        wt = up(torch.rand(R, 1, 1, 32) * 0.1, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        # per-row valid counts: the users at ~half the slab, ragged
        nv = torch.tensor([N // 2 + 37 * i for i in range(R)], dtype=torch.float32).reshape(R, 1, 1, 1)
        st = {
            "nvalid": up(nv, ttnn.float32, ttnn.TILE_LAYOUT),
            "valid": up(torch.full((1, 1, 1, 1), int(nv.max()), dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
        }
        print(f"== N={N} rows/user={n} R={R}", flush=True)
        s = idx.score(q, wt)
        print(f"IDXB N={N} score          {traced_us(md, lambda: idx.score(q, wt)):9.1f} us", flush=True)
        ar = idx._arange_for(idx.n_score)
        print(
            f"IDXB N={N} mask_scores    {traced_us(md, lambda: I.mask_scores(s, st['nvalid'], ar)):9.1f} us", flush=True
        )
        ms = I.mask_scores(s, st["nvalid"], ar)
        print(
            f"IDXB N={N} topk(masked)   {traced_us(md, lambda: ttnn.experimental.topk_large_indices(ms, k=512, valid_length_tensor=st['valid'])):9.1f} us",
            flush=True,
        )
        print(f"IDXB N={N} select(all)    {traced_us(md, lambda: idx.select(s, st)):9.1f} us", flush=True)
        print(
            f"IDXB N={N} score+select   {traced_us(md, lambda: idx.select(idx.score(q, wt), st)):9.1f} us", flush=True
        )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384, "trace_region_size": 200_000_000}], indirect=True)
@torch.no_grad()
def test_idx_score_ops(mesh_device):
    """score stage split: matmul1 / matmul2 / slice / untilize, and the block-diagonal one-matmul second stage."""
    md = mesh_device
    U, n = int(os.environ.get("IDXB_U", "1")), NROW
    R = U * n
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t, dt, lay: ttnn.from_torch(t.contiguous(), device=md, dtype=dt, layout=lay, mesh_mapper=rep)
    ckc = ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    for N in NS:
        kc = up(torch.randn(U, 1, N, 128), ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
        q = up(torch.randn(U, 1, 32 * n, 128), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        w = up(torch.rand(R, 1, 1, 32) * 0.1, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        s1 = ttnn.matmul(q, kc, transpose_b=True, activation="relu", compute_kernel_config=ckc)
        s1r = ttnn.reshape(s1, (R, 1, 32, N))
        print(f"== N={N}", flush=True)
        print(
            f"IDXS N={N} mm1                {traced_us(md, lambda: ttnn.matmul(q, kc, transpose_b=True, activation='relu', compute_kernel_config=ckc)):9.1f} us",
            flush=True,
        )
        print(
            f"IDXS N={N} mm2 (R batch)      {traced_us(md, lambda: ttnn.matmul(w, s1r, compute_kernel_config=ckc)):9.1f} us",
            flush=True,
        )
        s2 = ttnn.matmul(w, s1r, compute_kernel_config=ckc)
        print(
            f"IDXS N={N} slice+untilize     {traced_us(md, lambda: ttnn.to_layout(ttnn.slice(s2, [0, 0, 0, 0], [R, 1, 1, N]), ttnn.ROW_MAJOR_LAYOUT)):9.1f} us",
            flush=True,
        )
        # block-diagonal second stage: [1,1,R,32R] @ [1,1,32R,N]
        sel = torch.zeros(R, 32 * R)
        for r in range(R):
            sel[r, r * 32 : (r + 1) * 32] = 1.0
        selt = up(sel.reshape(1, 1, R, 32 * R), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        wf = up(torch.rand(1, 1, R, 32) * 0.1, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        s1v = ttnn.reshape(s1, (1, 1, R * 32, N))

        def bd():
            wbd = ttnn.multiply(ttnn.repeat(wf, ttnn.Shape([1, 1, 1, R])), selt)
            return ttnn.matmul(wbd, s1v, compute_kernel_config=ckc)

        print(f"IDXS N={N} blockdiag(w+mm2)   {traced_us(md, bd):9.1f} us", flush=True)
        wbd = ttnn.multiply(ttnn.repeat(wf, ttnn.Shape([1, 1, 1, R])), selt)
        print(
            f"IDXS N={N} blockdiag mm2 only {traced_us(md, lambda: ttnn.matmul(wbd, s1v, compute_kernel_config=ckc)):9.1f} us",
            flush=True,
        )
        s2b = ttnn.matmul(wbd, s1v, compute_kernel_config=ckc)
        print(
            f"IDXS N={N} untilize [1,1,R,N] {traced_us(md, lambda: ttnn.to_layout(s2b, ttnn.ROW_MAJOR_LAYOUT)):9.1f} us",
            flush=True,
        )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384, "trace_region_size": 200_000_000}], indirect=True)
@torch.no_grad()
def test_idx_mm1_variants(mesh_device):
    md = mesh_device
    U, n = int(os.environ.get("IDXB_U", "1")), NROW
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t, dt, lay: ttnn.from_torch(t.contiguous(), device=md, dtype=dt, layout=lay, mesh_mapper=rep)
    mk = lambda fid, acc=True: ttnn.init_device_compute_kernel_config(
        md.arch(), math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=acc, packer_l1_acc=False
    )
    ck4 = mk(ttnn.MathFidelity.HiFi4)
    for N in NS:
        kt = torch.randn(U, 1, N, 128)
        kc = up(kt, ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
        q = up(torch.randn(U, 1, 32 * n, 128), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        ref = ttnn.to_torch(ttnn.matmul(q, kc, transpose_b=True, activation="relu", compute_kernel_config=ck4))
        print(f"== N={N}", flush=True)
        for name, fid in (
            ("HiFi4", ttnn.MathFidelity.HiFi4),
            ("HiFi2", ttnn.MathFidelity.HiFi2),
            ("LoFi", ttnn.MathFidelity.LoFi),
        ):
            ck = mk(fid)
            f = lambda: ttnn.matmul(q, kc, transpose_b=True, activation="relu", compute_kernel_config=ck)
            o = ttnn.to_torch(f())
            print(
                f"IDXV N={N} mm1 {name}: {traced_us(md, f):8.1f} us  bit-equal to HiFi4: {bool(torch.equal(o, ref))}",
                flush=True,
            )
        # per-user 2D matmuls on separate key tensors
        kus = [up(kt[u : u + 1], ttnn.bfloat8_b, ttnn.TILE_LAYOUT) for u in range(U)]
        qus = [ttnn.slice(q, [u, 0, 0, 0], [u + 1, 1, 32 * n, 128]) for u in range(U)]
        f = lambda: [
            ttnn.matmul(qus[u], kus[u], transpose_b=True, activation="relu", compute_kernel_config=ck4)
            for u in range(U)
        ]
        o = torch.cat([ttnn.to_torch(x) for x in f()], dim=0)
        print(
            f"IDXV N={N} mm1 per-user x4: {traced_us(md, f):8.1f} us  bit-equal: {bool(torch.equal(o, ref))}",
            flush=True,
        )
        # no relu fused / bf16 key
        f = lambda: ttnn.matmul(q, kc, transpose_b=True, compute_kernel_config=ck4)
        print(f"IDXV N={N} mm1 no relu: {traced_us(md, f):8.1f} us", flush=True)
        kb = up(kt, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        f = lambda: ttnn.matmul(q, kb, transpose_b=True, activation="relu", compute_kernel_config=ck4)
        print(f"IDXV N={N} mm1 bf16 keys: {traced_us(md, f):8.1f} us", flush=True)
        # pre-transposed key slab [U,1,128,N]
        ktr = up(kt.transpose(2, 3), ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
        f = lambda: ttnn.matmul(q, ktr, activation="relu", compute_kernel_config=ck4)
        o = ttnn.to_torch(f())
        print(
            f"IDXV N={N} mm1 K^T stored: {traced_us(md, f):8.1f} us  bit-equal: {bool(torch.equal(o, ref))}", flush=True
        )
