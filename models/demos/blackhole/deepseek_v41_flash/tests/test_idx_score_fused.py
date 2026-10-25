# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt/idx_score.py score_mask_fused against the chain (SpecIndexer.score + mask_scores): bit-exact, and its time. One chip, U users x n rows."""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_idx_lever_bench import traced_us
from models.demos.blackhole.deepseek_v41_flash.tt import indexer as I
from models.demos.blackhole.deepseek_v41_flash.tt.idx_score import score_mask_fused
from models.demos.blackhole.deepseek_v41_flash.tt.spec_paged import SpecIndexer

NS = [int(x) for x in os.environ.get("IDXB_N", "2080,32832,65568").split(",")]


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384, "trace_region_size": 200_000_000}], indirect=True)
@pytest.mark.parametrize("U,n", [(1, 4), (1, 6), (4, 4)])
@torch.no_grad()
def test_fused(mesh_device, U, n):
    md = mesh_device
    R = U * n
    torch.manual_seed(1)
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t, dt, lay: ttnn.from_torch(t.contiguous(), device=md, dtype=dt, layout=lay, mesh_mapper=rep)
    for N in NS:
        for kdt in (ttnn.bfloat8_b, ttnn.bfloat16):
            w = {"wq_b": torch.randn(4096, 1280) * 0.02, "weights_proj": torch.randn(32, 5120) * 0.02}
            freqs = torch.polar(torch.ones(N * 2 + 64, 32), torch.rand(N * 2 + 64, 32))
            idx = SpecIndexer(
                md, w, freqs, users_per_row=U, n_alloc=N, ratio=2, key_dtype=kdt, fp4_q=True, backend="matmul", n=n
            )
            idx.load_keys(torch.randn(U, N - 32, 128))
            nb = N - 32 if N > 4096 else N
            idx.nb = nb
            q = up(torch.randn(U, 1, 32 * n, 128), ttnn.bfloat16, ttnn.TILE_LAYOUT)
            wflat = torch.randn(1, 1, R, 32) * 0.1  # signed gates
            wt = up(wflat, ttnn.bfloat16, ttnn.TILE_LAYOUT)
            w4 = up(wflat.reshape(R, 1, 1, 32), ttnn.bfloat16, ttnn.TILE_LAYOUT)
            nv = torch.tensor([nb // 2 + 3 * i for i in range(R)], dtype=torch.float32)
            nvt = up(nv.reshape(R, 1, 1, 1), ttnn.float32, ttnn.TILE_LAYOUT)
            nvu = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
            nvu[0, 0, 0, :R] = nv.to(torch.int32)
            nvu_d = up(nvu, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            chain = lambda: I.mask_scores(idx.score(q, w4), nvt, idx._arange_for(idx.n_score))
            ref = ttnn.to_torch(chain())[:, :, :R, :].reshape(R, -1)
            vd = up(torch.full((1, 1, 1, 1), nb, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            fused = lambda: score_mask_fused(q, idx.k_cache, wt, nvu_d, n, nb, valid=vd)
            got = ttnn.to_torch(fused()).reshape(R, -1)
            eq = torch.equal(got, ref)
            nd = int((got != ref).sum())
            print(
                f"FUSED U={U} n={n} N={nb} key={kdt}: bit-exact {eq} (diff {nd}/{got.numel()})  chain {traced_us(md, chain):.1f} us  fused {traced_us(md, fused):.1f} us",
                flush=True,
            )
            if not eq:
                bad = (got != ref).nonzero()[:5]
                for b in bad:
                    print("   ", b.tolist(), float(got[tuple(b)]), float(ref[tuple(b)]))
            assert eq
