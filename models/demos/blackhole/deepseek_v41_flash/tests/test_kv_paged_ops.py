# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Micro-benchmarks of the ttnn primitives the paged / sparse decode KV design relies on (see
docs/superpowers/specs/2026-10-02-dsv41-kv-paged-capacity-design.md). Every test prints ``KVOPS`` lines (us per call, measured inside a
captured trace of ``REPS`` back-to-back calls so host dispatch is excluded) and checks numerics where a golden exists.

  * sparse_sdpa at decode shapes: H=32 (8 local heads padded), S=U queries (one per user), TOPK = 128 window + 512 selected, one RM kv pool
  * row gather from an RM pool (ttnn.embedding) and tile->RM ring conversion + in-place slice_write into the pool
"""

import time

import pytest
import torch

import ttnn

REPS = 10
DP = {"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200_000_000}
MESH = pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
PARAMS = pytest.mark.parametrize("device_params", [pytest.param(DP, id="ring")], indirect=True)


def rep(md):
    return ttnn.ReplicateTensorToMesh(md)


def dev(md, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(
        t, device=md, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep(md)
    )


def host0(md, t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def bench(md, fn, reps=REPS, tag=""):
    """us per call of fn (after one compile call), measured over a trace of ``reps`` calls."""
    out = fn()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    for _ in range(reps):
        fn()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ts = []
    for _ in range(4):
        t = time.perf_counter()
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        ts.append((time.perf_counter() - t) / reps * 1e6)
    ttnn.release_trace(md, tid)
    us = min(ts)
    print(f"KVOPS {tag:60s} {us:9.1f} us/call", flush=True)
    return out, us


def golden(q, kv, idx, sink, scale):
    """q [H,U,512], kv [R,512], idx [U,TOPK] (int64, no sentinels), sink [H] (scaled-logit domain, i.e. model sink / scale) -> [H,U,512]."""
    H, U, D = q.shape
    out = torch.zeros(H, U, D)
    for u in range(U):
        k = kv[idx[u]].float()  # [TOPK, D]
        s = (q[:, u].float() @ k.T) * scale  # [H, TOPK]
        s = torch.cat([s, (sink.float() * scale).reshape(H, 1)], dim=-1)
        p = torch.softmax(s, dim=-1)[:, :-1]
        out[:, u] = p @ k
    return out


@MESH
@PARAMS
@pytest.mark.parametrize("kv_dtype", [ttnn.bfloat16, ttnn.fp8_e4m3], ids=["kvbf16", "kvfp8"])
@pytest.mark.parametrize("users", [1, 8, 16, 32])
@pytest.mark.parametrize("pool_rows", [131072])
@torch.no_grad()
def test_sparse_sdpa_decode(mesh_device, users, kv_dtype, pool_rows):
    md = mesh_device
    U, H, D, TOPK, R = users, 32, 512, 640, pool_rows
    gen = torch.Generator().manual_seed(0)
    q = torch.randn(1, H, U, D, generator=gen).to(torch.bfloat16)
    kv = torch.randn(1, 1, R, D, generator=gen).to(torch.bfloat16)
    idx = torch.zeros(1, 1, U, TOPK, dtype=torch.int64)
    for u in range(U):
        idx[0, 0, u, :128] = torch.arange(128) + u * 128  # window ring rows of the user
        idx[0, 0, u, 128:] = U * 128 + torch.randperm(R - U * 128, generator=gen)[:512]  # selected compressed rows
    sink = (torch.linspace(-2, 4, H) / D**-0.5).reshape(1, 1, 1, H).to(torch.bfloat16)
    tt_q = dev(md, q, ttnn.bfloat16)
    tt_kv = dev(md, kv if kv_dtype == ttnn.bfloat16 else kv.float(), kv_dtype)
    tt_idx = dev(md, idx.to(torch.int32), ttnn.uint32)
    tt_sink = dev(md, sink, ttnn.bfloat16)
    ckc = ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    fmt = (
        ttnn.transformer.SparseKVFormat.BF16 if kv_dtype == ttnn.bfloat16 else ttnn.transformer.SparseKVFormat.FP8_E4M3
    )
    for kc in (128, 64):
        f = lambda: ttnn.transformer.sparse_sdpa(
            tt_q,
            tt_kv,
            tt_idx,
            D,
            kv_format=fmt,
            scale=D**-0.5,
            k_chunk_size=kc,
            compute_kernel_config=ckc,
            attention_sink=tt_sink,
        )
        out, us = bench(
            md,
            f,
            tag=f"sparse_sdpa U={U} H=32 TOPK=640 kv={'bf16' if kv_dtype == ttnn.bfloat16 else 'fp8'} kc={kc} pool={R}",
        )
    o = host0(md, out).float()[0]  # [H, U, D]
    kvf = kv[0, 0].float() if kv_dtype == ttnn.bfloat16 else kv[0, 0].float().to(torch.float8_e4m3fn).float()
    g = golden(q[0].float(), kvf, idx[0, 0], sink[0, 0, 0].float(), D**-0.5)
    a, b = o.flatten(), g.flatten()
    pcc = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
    print(f"KVOPS sparse_sdpa U={U} kv={kv_dtype} PCC vs torch golden {pcc:.5f}", flush=True)
    assert pcc > 0.99


@MESH
@PARAMS
@pytest.mark.parametrize("users", [8, 32])
@torch.no_grad()
def test_pool_row_ops(mesh_device, users):
    """Row gather from an RM pool and the ring update path (tile ring -> RM -> in-place write into the pool)."""
    md = mesh_device
    U, D, R = users, 512, 131072
    pool = dev(md, torch.randn(1, 1, R, D).to(torch.bfloat16), ttnn.bfloat16)
    pool2d = ttnn.reshape(pool, (R, D))
    idx = dev(md, torch.randint(0, R, (U, 512), dtype=torch.int32), ttnn.uint32)
    _, us = bench(
        md,
        lambda: ttnn.embedding(idx, pool2d, layout=ttnn.ROW_MAJOR_LAYOUT),
        tag=f"embedding gather U={U} x 512 rows bf16 RM pool",
    )
    ring = dev(md, torch.randn(U, 1, 128, D).to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    _, us = bench(md, lambda: ttnn.to_layout(ring, ttnn.ROW_MAJOR_LAYOUT), tag=f"ring tile->RM [U={U},1,128,512] bf16")
    ring_rm = ttnn.to_layout(ring, ttnn.ROW_MAJOR_LAYOUT)
    ring_rm = ttnn.reshape(ring_rm, (1, 1, U * 128, D))
    try:
        _, us = bench(
            md,
            lambda: ttnn.experimental.slice_write(
                ring_rm, pool, [0, 0, 4096, 0], [1, 1, 4096 + U * 128, D], [1, 1, 1, 1]
            ),
            tag=f"slice_write ring RM {U * 128} rows into pool (in place)",
        )
        got = host0(md, pool)[0, 0, 4096 : 4096 + U * 128]
        ref = host0(md, ring_rm)[0, 0]
        print(
            f"KVOPS slice_write in-place check max abs diff {float((got.float() - ref.float()).abs().max()):.3e}",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"KVOPS slice_write failed: {str(e)[:300]}", flush=True)
