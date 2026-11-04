# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""otherlayers: unit tests (one device) of the decode-indexer / owner-layer ops: bit equality against the stock op and traced device time (difference of two trace lengths)."""

import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import indexer as IX
from models.demos.blackhole.deepseek_v41_flash.tt.attention import _cfg_1d

TRACE = [{"trace_region_size": 20000000}]


def traced_us(device, fn, k=8, reps=20):
    def cap(kk):
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        for _ in range(kk):
            fn()
        ttnn.end_trace_capture(device, tid, cq_id=0)
        return tid

    def run(tid):
        for _ in range(3):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        return (time.perf_counter() - t0) / reps * 1e6

    t1, tk = cap(1), cap(k)
    return (run(tk) - run(t1)) / (k - 1)


@pytest.mark.parametrize("device_params", TRACE, indirect=True)
@pytest.mark.parametrize("R", [8, 12])
def test_wq_b_cfg(device, R):
    """indexer wq_b linear [R,1280] x [1280,4096] bfp8: core_grid (2,8) (today) vs the attention 1D configs"""
    torch.manual_seed(0)
    x = ttnn.from_torch(
        torch.randn(1, 1, R, 1280).to(torch.bfloat16), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    w = ttnn.from_torch(
        (torch.randn(1, 1, 1280, 4096) * 0.03).float(), device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
    )
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    base = lambda: ttnn.linear(x, w, compute_kernel_config=ckc, core_grid=ttnn.CoreGrid(y=2, x=8))
    ref = ttnn.to_torch(base())
    print(f"wq_b R={R} core_grid(2,8): {traced_us(device, base):.1f} us", flush=True)
    for mc in (16, 32, 64):
        pc = _cfg_1d(1280, 4096, mc)
        f = lambda: ttnn.linear(x, w, compute_kernel_config=ckc, program_config=pc)
        got = ttnn.to_torch(f())
        eq = bool((got == ref).all())
        print(
            f"wq_b R={R} 1d:{mc}: {traced_us(device, f):.1f} us, bit-equal {eq}, maxabs {float((got.float() - ref.float()).abs().max()):.3g}",
            flush=True,
        )


@pytest.mark.parametrize("device_params", TRACE, indirect=True)
@pytest.mark.parametrize("R,max_id", [(8, 4128), (8, 8224), (12, 16480), (8, 1056)])
def test_ids_sort_bm(device, R, max_id):
    """bitmap id sort vs the radix sort (distinct ids per row, some rows with 0xFFFFFFFF sentinels)"""
    torch.manual_seed(1)
    K = 512
    rows = []
    for r in range(R):
        n = K if r % 3 else 400
        ids = torch.randperm(max_id)[:n].to(torch.int64)
        if n < K:
            ids = torch.cat([ids, torch.full((K - n,), 0xFFFFFFFF, dtype=torch.int64)])
        rows.append(ids)
    ids = torch.stack(rows).to(torch.int64).reshape(R, 1, 1, K)
    t = ttnn.from_torch(
        ids.to(torch.int32).view(torch.uint32) if False else ids.to(torch.int32),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    os.environ["DSV41_OL_IDS_BM"] = "0"
    ref = ttnn.to_torch(IX.ids_sort(t, max_id))
    os.environ["DSV41_OL_IDS_BM"] = "1"
    got = ttnn.to_torch(IX.ids_sort(t, max_id))
    eq = bool((got.to(torch.int64) & 0xFFFFFFFF == ref.to(torch.int64) & 0xFFFFFFFF).all())
    print(f"ids_sort R={R} max_id={max_id}: bit-equal {eq}", flush=True)
    for mode in ("0", "1"):
        os.environ["DSV41_OL_IDS_BM"] = mode
        print(
            f"ids_sort R={R} max_id={max_id} bm={mode}: {traced_us(device, lambda: IX.ids_sort(t, max_id)):.1f} us",
            flush=True,
        )
    os.environ["DSV41_OL_IDS_BM"] = "0"
    assert eq


@pytest.mark.parametrize("device_params", TRACE, indirect=True)
@pytest.mark.parametrize("U,n", [(2, 4), (2, 6), (1, 4), (4, 2)])
def test_key_rows(device, U, n):
    """key_rows (fp4 + per-block-index update tiles in one program) vs pad / fp4_fused / slice / to_layout(RM) / per-j slice, to_layout(TILE), to_memory_config(kcfg)"""
    from models.demos.blackhole.deepseek_v41_flash.tt import ol_idx

    R = U * n
    torch.manual_seed(5)
    kk = (torch.randn(1, 1, R, 128) * torch.logspace(-2, 2, R).reshape(1, 1, R, 1)).to(torch.bfloat16)
    kk[0, 0, 1, :32] = 0
    k = ttnn.from_torch(kk, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    kcfg = ttnn.create_sharded_memory_config(
        shape=(32, 128),
        core_grid=ttnn.num_cores_to_corerangeset(U, ttnn.CoreCoord(8, 8), row_wise=True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    def stock():
        kp = IX.DSV41DecodeIndexer._fp4_fused_rows(k)
        k_rm = ttnn.reshape(ttnn.to_layout(ttnn.reshape(kp, (1, 1, R, 128)), ttnn.ROW_MAJOR_LAYOUT), (U, n, 128))
        outs = []
        for j in range(n):
            kj = ttnn.slice(k_rm, [0, j, 0], [U, j + 1, 128])
            kj = ttnn.to_layout(ttnn.reshape(kj, (1, U, 1, 128)), ttnn.TILE_LAYOUT)
            outs.append(ttnn.to_memory_config(kj, kcfg))
        return outs

    fused = lambda: ol_idx.key_rows(k, U, n, kcfg)
    a, b = stock(), fused()
    bad = 0
    for j in range(n):
        x, y = ttnn.to_torch(a[j]), ttnn.to_torch(b[j])
        bad += int((x != y).sum())
    print(f"key_rows U={U} n={n}: {bad} elements differ", flush=True)
    print(
        f"key_rows U={U} n={n}: stock {traced_us(device, stock):.1f} us, fused {traced_us(device, fused):.1f} us",
        flush=True,
    )
    assert bad == 0


@pytest.mark.parametrize("device_params", TRACE, indirect=True)
@pytest.mark.parametrize("R,n", [(8, 8256), (12, 8256), (8, 4128), (8, 32800)])
def test_cs_cand_fast(device, R, n):
    """cs_cand_fast (whole-row read, 4 cores per row) vs the stock candidate kernel"""
    if os.environ.get("OLS_EXP") != "1":
        pytest.skip("experimental (OLS_EXP=1)")
    from models.demos.blackhole.deepseek_v41_flash.tt import indexer_colshard as CS
    from models.demos.blackhole.deepseek_v41_flash.tt import ol_idx

    torch.manual_seed(6)
    K = 512
    sc = ttnn.from_torch(
        torch.randn(R, 1, 1, n).to(torch.bfloat16), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    ids = torch.stack([torch.randperm(n)[:K] for _ in range(R)]).reshape(R, 1, 1, K).to(torch.int32)
    ids[1, 0, 0, 5] = n + 7  # a sentinel-like id
    ids[2, 0, 0, 9] = -1  # 0xFFFFFFFF
    lids = ttnn.from_torch(ids, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    colid = ttnn.from_torch(
        torch.full((1, 1, 1, 16), 3, dtype=torch.int32), device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    os.environ["DSV41_OL_CSCAND"] = "0"
    v0, g0 = CS.cs_cand(sc, lids, colid, 8)
    v1, g1 = ol_idx.cs_cand_fast(sc, lids, colid, 8, split=4)
    a, b = ttnn.to_torch(v0).float(), ttnn.to_torch(v1).float()
    ga, gb = ttnn.to_torch(g0).to(torch.int64) & 0xFFFFFFFF, ttnn.to_torch(g1).to(torch.int64) & 0xFFFFFFFF
    nd = int((a != b).sum()) + int((ga != gb).sum())
    print(f"cs_cand R={R} n={n}: {nd} elements differ", flush=True)
    print(
        f"cs_cand R={R} n={n}: stock {traced_us(device, lambda: CS.cs_cand(sc, lids, colid, 8)):.1f} us, fast {traced_us(device, lambda: ol_idx.cs_cand_fast(sc, lids, colid, 8, split=4)):.1f} us",
        flush=True,
    )
    assert nd == 0


def _rope_tables(R, seed=3):
    g = torch.Generator().manual_seed(seed)
    ang = torch.rand(R, 32, generator=g) * 6.28
    cos, sin = torch.cos(ang).repeat_interleave(2, dim=-1), torch.sin(ang).repeat_interleave(2, dim=-1)
    C, S = torch.ones(R, 128), torch.zeros(R, 128)
    C[:, 64:], S[:, 64:] = cos, sin
    return C.reshape(R, 1, 1, 128).to(torch.bfloat16), S.reshape(R, 1, 1, 128).to(torch.bfloat16)


@pytest.mark.parametrize("device_params", TRACE, indirect=True)
@pytest.mark.parametrize("R", [8, 12])
def test_idx_qpost(device, R):
    """idx_qpost (reshape + RoPE in one program) vs the stock chain reshape / multiply / matmul(P) / addcmul"""
    from models.demos.blackhole.deepseek_v41_flash.tt import ol_idx

    torch.manual_seed(4)
    ql = (torch.randn(1, 1, R, 4096) * 2.0).to(torch.bfloat16)
    ql[0, 0, 0, 5] = 0.0
    ql[0, 0, 1, 70] = -0.0
    C, S = _rope_tables(R)
    up = lambda t: ttnn.from_torch(t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    qlin, Ct, St = up(ql), up(C), up(S)
    P = up(IX._pair_swap_matrix().reshape(1, 1, 128, 128).to(torch.bfloat16))
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    def stock():
        q = ttnn.reshape(qlin, (R, 1, 32, 128))
        return ttnn.addcmul(ttnn.multiply(q, Ct), ttnn.matmul(q, P, compute_kernel_config=ckc), St)

    qlin_l1 = ttnn.to_memory_config(qlin, ttnn.L1_MEMORY_CONFIG)
    a = ttnn.to_torch(stock())
    print(f"qpost R={R}: stock {traced_us(device, stock):.1f} us", flush=True)
    bad = 0
    for v2 in ("0", "1") if os.environ.get("OLS_QPOST2") == "1" else ("0",):
        os.environ["DSV41_OL_QPOST2"] = v2
        for tag, ql in (("dram", qlin),) + ((("l1", qlin_l1),) if os.environ.get("OLS_DIRECT") == "1" else ()):
            fused = lambda: ol_idx.idx_qpost(ql, Ct, St, R, out_l1=True)
            b = ttnn.to_torch(fused())
            ndiff = int((a != b).sum())
            bad += ndiff
            print(
                f"qpost R={R} v2={v2} qlin {tag}: {ndiff} of {a.numel()} elements differ, maxabs {float((a.float() - b.float()).abs().max()):.3g}, fused {traced_us(device, fused):.1f} us",
                flush=True,
            )
    os.environ["DSV41_OL_QPOST2"] = "0"
    assert bad == 0
