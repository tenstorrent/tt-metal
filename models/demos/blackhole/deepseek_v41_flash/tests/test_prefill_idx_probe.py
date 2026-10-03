# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Probe of the ops the PREFILL indexer path is built from (random data, no checkpoint): multi-query ``indexer_score_dsa`` with a causal chunk
start, ``topk_large_indices`` on the [Sq, T] score rows, ``slice_write`` into tile / row-major slabs, ``sparse_sdpa`` with many queries and
``cache_batch_idx``. Prints ``PIDX`` lines."""

import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference.ref_layer import pcc

DP = {"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200_000_000}
MESH = pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
PARAMS = pytest.mark.parametrize("device_params", [pytest.param(DP, id="ring")], indirect=True)


def up(md, t, dtype, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t.contiguous(),
        device=md,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )


def d0(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def timed(md, fn, n=3):
    fn()
    ttnn.synchronize_device(md)
    t = time.perf_counter()
    for _ in range(n):
        fn()
    ttnn.synchronize_device(md)
    return (time.perf_counter() - t) / n * 1e3


@MESH
@PARAMS
@torch.no_grad()
def test_score_topk(mesh_device):
    md = mesh_device
    H, D = 32, 128
    gen = torch.Generator().manual_seed(0)
    for Sq, T, cs in ((128, 1024, 896), (256, 2048, 1792), (256, 2048, 2048 - 32)):
        k = torch.randn(1, 1, T, D, generator=gen).to(torch.bfloat16)
        q = torch.randn(1, H, Sq, D, generator=gen).to(torch.bfloat16)
        w = (torch.randn(1, 1, Sq, H, generator=gen) * 0.1).to(torch.bfloat16)
        sc = ttnn.experimental.indexer_score_dsa(
            up(md, q, ttnn.bfloat16),
            up(md, k, ttnn.bfloat16),
            up(md, w, ttnn.bfloat16),
            chunk_start_idx=cs,
            kv_len=T,
            program_config=ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=128, head_group_size=0),
        )
        got = d0(sc).float()
        ref = torch.zeros(Sq, T)
        for h in range(H):
            ref += torch.relu(q[0, h].float() @ k[0, 0].float().T) * w[0, 0, :, h : h + 1].float()
        fut = torch.arange(T).unsqueeze(0) > cs + torch.arange(Sq).unsqueeze(1)
        ref = ref.masked_fill(fut, float("-inf"))
        fin = torch.isfinite(ref)
        print(
            f"PIDX score Sq={Sq} T={T} cs={cs}: shape {tuple(sc.shape)} {sc.layout} {sc.dtype}; finite-mask equal {bool((torch.isfinite(got.reshape(Sq,T)) == fin).all())}; "
            f"PCC {pcc(got.reshape(Sq, T)[fin], ref[fin]):.5f}",
            flush=True,
        )
        ids = d0(ttnn.experimental.topk_large_indices(sc, k=512)).reshape(Sq, 512).long()
        ok = []
        for s in (0, 1, Sq // 2, Sq - 1):
            nvis = int(fin[s].sum())
            sel = set(ids[s][ids[s] != 0xFFFFFFFF].tolist())
            exp = set(ref[s].topk(min(512, nvis)).indices.tolist())
            ok.append((nvis, len(sel), len(sel & exp), int((ids[s] == 0xFFFFFFFF).sum())))
        print(f"PIDX topk (n_visible, n_selected, n_correct, n_skip) rows 0,1,mid,last: {ok}", flush=True)
        qd, kd, wd = up(md, q, ttnn.bfloat16), up(md, k, ttnn.bfloat16), up(md, w, ttnn.bfloat16)
        cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=128, head_group_size=0)
        tm = timed(
            md,
            lambda: ttnn.experimental.indexer_score_dsa(qd, kd, wd, chunk_start_idx=cs, kv_len=T, program_config=cfg),
        )
        print(f"PIDX score time {tm:.2f} ms", flush=True)


@MESH
@PARAMS
@torch.no_grad()
def test_slice_write(mesh_device):
    md = mesh_device
    gen = torch.Generator().manual_seed(0)
    U, T, D = 2, 256, 128
    slab = up(md, torch.zeros(U, 1, T, D).to(torch.bfloat16), ttnn.bfloat16)
    new = torch.randn(U, 1, 64, D, generator=gen).to(torch.bfloat16)
    try:
        ttnn.experimental.slice_write(up(md, new, ttnn.bfloat16), slab, [0, 0, 128, 0], [U, 1, 192, D], [1, 1, 1, 1])
        got = d0(slab).float()
        print(
            f"PIDX slice_write tile bf16 ok: written PCC {pcc(got[:, :, 128:192], new.float()):.5f}, rest zero {bool((got[:, :, :128] == 0).all() and (got[:, :, 192:] == 0).all())}",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"PIDX slice_write tile FAILED {str(e)[:300]}", flush=True)
    rm = up(md, torch.zeros(1, 1, 512, 512).to(torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    new = torch.randn(1, 1, 64, 512, generator=gen).to(torch.bfloat16)
    try:
        ttnn.experimental.slice_write(
            up(md, new, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT), rm, [0, 0, 100, 0], [1, 1, 164, 512], [1, 1, 1, 1]
        )
        got = d0(rm).float()
        print(f"PIDX slice_write RM ok: PCC {pcc(got[:, :, 100:164], new.float()):.5f}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"PIDX slice_write RM FAILED {str(e)[:300]}", flush=True)
    x = up(
        md,
        torch.arange(64).reshape(1, 1, 64, 1).expand(1, 1, 64, 128).contiguous().to(torch.bfloat16),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    )
    try:
        y = ttnn.slice(x, [0, 0, 0, 0], [1, 1, 64, 128], [1, 1, 2, 1])
        print(f"PIDX strided slice RM: shape {tuple(y.shape)} rows {d0(y).float()[0, 0, :4, 0].tolist()}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"PIDX strided slice RM FAILED {str(e)[:300]}", flush=True)


@MESH
@PARAMS
@torch.no_grad()
def test_sparse_many_queries(mesh_device):
    md = mesh_device
    gen = torch.Generator().manual_seed(0)
    U, T, C, K = 2, 4096, 256, 640
    H = 32
    kv = (torch.randn(U, 1, T, 512, generator=gen) * 0.5).to(torch.bfloat16)
    q = (torch.randn(1, H, C, 512, generator=gen) * 0.5).to(torch.bfloat16)
    idx = torch.randint(0, T, (1, 1, C, K), generator=gen)
    idx[..., 600:] = 0xFFFFFFFF
    sink = torch.zeros(1, 1, 1, H).to(torch.bfloat16)
    kvd, qd = up(md, kv, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT), up(md, q, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    idd = up(md, idx.to(torch.int64).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    sk = up(md, sink, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    ckc = ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    def run(u):
        return ttnn.transformer.sparse_sdpa(
            qd,
            kvd,
            idd,
            512,
            kv_format=ttnn.transformer.SparseKVFormat.BF16,
            scale=512**-0.5,
            k_chunk_size=128,
            compute_kernel_config=ckc,
            attention_sink=sk,
            cache_batch_idx=u,
        )

    for u in range(U):
        o = d0(run(u)).float().reshape(H, C, 512)
        ref = torch.zeros(H, C, 512)
        for s in (0, 7, C - 1):
            ii = idx[0, 0, s][idx[0, 0, s] != 0xFFFFFFFF].long()
            Kr = kv[u, 0, ii].float()
            sc = (q[0, :, s].float() @ Kr.T) * 512**-0.5
            p = torch.softmax(torch.cat([sc, torch.zeros(H, 1)], -1), -1)[:, :-1]
            ref[:, s] = p @ Kr
        print(
            f"PIDX sparse_sdpa C={C} H={H} user {u}: PCC rows 0/7/last {[round(pcc(o[:, s], ref[:, s]), 5) for s in (0, 7, C - 1)]}",
            flush=True,
        )
    print(f"PIDX sparse_sdpa C={C} H=32 K=640: {timed(md, lambda: run(0)):.2f} ms/call", flush=True)
