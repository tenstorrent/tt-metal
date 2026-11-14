"""verify-rows: the decode indexer score + select at S rows is bit-identical to S=1 run row by row (same inputs), causal per row. 1x8 ring slot. Run: vr.sh <tag> tests.test_hy4_vrows_score_device MESH_DEVICE=1x8 HY4_FABRIC=ring NS=65536 SS=2,4,8"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tt"))
from dm_block import close_row, open_row

import ttnn
from models.demos.blackhole.hy4_preview.tt import hy4_idx_score as IS
from models.demos.blackhole.hy4_preview.tt import hy4_pool as HP
from models.demos.blackhole.hy4_preview.tt.hy4_pool_decode import PoolCtx, select_global
from models.demos.gpt_oss.config import mesh_1x8
from models.demos.gpt_oss.tt.ccl import CCLManager

K = 2048


def emit(**kw):
    print("RES " + json.dumps(kw), flush=True)


def rep(md, t, dt, lay):
    return ttnn.from_torch(
        t,
        device=md,
        dtype=dt,
        layout=lay,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def host0(md, t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def main():
    full, md = open_row()
    try:
        mc = mesh_1x8()
        ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
        g = torch.Generator().manual_seed(3)
        for N in [int(x) for x in os.environ.get("NS", "65536").split(",")]:
            pages = N // 512
            kp = ttnn.empty(
                [pages, 1, 64, 128],
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=md,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            kview = ttnn.reshape(kp, [1, 1, pages * 64, 128])
            ctx1 = PoolCtx(md, mc, ccl, pages, rows=1, S=1, mode="topk", bucket_tokens=N, fuse="sparse")
            nch = min(N, 65536)
            keys = torch.randn(nch, 128, generator=g).bfloat16()
            kb = ttnn.typecast(rep(md, keys.reshape(1, 1, nch, 128), ttnn.bfloat16, ttnn.TILE_LAYOUT), ttnn.bfloat8_b)
            for c0 in range(0, N, nch):
                HP.slice_copy(kb, kview, ctx1.colid, 8, HP.KEYTILE, nch // 512, dst_page0=c0 // 512)
            ttnn.synchronize_device(md)
            for S in [int(x) for x in os.environ.get("SS", "2,4,8").split(",")]:
                P0 = N - S - 3
                ctx = PoolCtx(md, mc, ccl, pages, rows=1, S=S, mode="topk", bucket_tokens=N, fuse="sparse")
                ctx.set_step([P0])
                qi = torch.randn(S, 1, 32, 128, generator=g).bfloat16()
                wi = torch.zeros(1, 1, 32, 32)
                wi[0, 0, :S] = torch.rand(S, 32, generator=g) / 16 + 1.0 / 64
                qh = rep(md, qi, ttnn.bfloat16, ttnn.TILE_LAYOUT)
                w = rep(md, wi.bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
                sc = IS.idx_score(qh, kview, w, 0, S, kvt=ctx.kvcnt, n_rows=ctx.n_local, kvt_rows=True)
                scS = host0(md, sc).reshape(S, -1)
                gS = host0(md, select_global(ctx, qh, w, kview)).reshape(S, K)
                ok_sc, ok_ids = True, True
                for j in range(S):
                    ctx1.set_step([P0 + j])
                    q1 = rep(md, qi[j : j + 1], ttnn.bfloat16, ttnn.TILE_LAYOUT)
                    w1 = torch.zeros(1, 1, 32, 32)
                    w1[0, 0, 0] = wi[0, 0, j]
                    w1 = rep(md, w1.bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
                    s1 = host0(md, IS.idx_score(q1, kview, w1, 0, 1, kvt=ctx1.kvcnt, n_rows=ctx1.n_local)).reshape(-1)
                    g1 = host0(md, select_global(ctx1, q1, w1, kview)).reshape(K)
                    ok_sc &= bool(torch.equal(scS[j].view(torch.int16), s1.view(torch.int16)))
                    ok_ids &= bool(
                        np.array_equal(
                            np.sort(gS[j].to(torch.int64).numpy() & 0xFFFFFFFF),
                            np.sort(g1.to(torch.int64).numpy() & 0xFFFFFFFF),
                        )
                    )
                emit(event="score_bit_identical", N=N, S=S, scores_equal_to_S1=ok_sc, ids_equal_to_S1=ok_ids)
    finally:
        close_row(full, md)


main()
