# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SpecIndexer.forward with the fused score+mask (DSV41_IDX_FUSED) against the chain: the top-512 ids must be IDENTICAL (same ids, same order), on random and tie-heavy (quantised) keys."""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_idx_lever_bench import traced_us
from models.demos.blackhole.deepseek_v41_flash.tt import spec_paged as SP
from models.demos.blackhole.deepseek_v41_flash.tt.spec_paged import SpecIndexer

NS = [int(x) for x in os.environ.get("IDXB_N", "4128,32832,65568").split(",")]


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384, "trace_region_size": 200_000_000}], indirect=True)
@pytest.mark.parametrize("U,n", [(1, 4), (1, 6), (4, 4)])
@torch.no_grad()
def test_fwd(mesh_device, U, n):
    md = mesh_device
    R = U * n
    torch.manual_seed(2)
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t, dt, lay: ttnn.from_torch(t.contiguous(), device=md, dtype=dt, layout=lay, mesh_mapper=rep)
    for N in NS:
        for tie in (False, True):
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
            if tie:
                keys = torch.round(keys * 1.5) / 1.5
            idx.load_keys(keys)
            idx.set_pos_bucket(None)
            x = up(torch.randn(1, 1, R, 5120), ttnn.bfloat16, ttnn.TILE_LAYOUT)
            qr = up(torch.randn(1, 1, R, 1280), ttnn.bfloat16, ttnn.TILE_LAYOUT)
            C = torch.ones(R, 1, 1, 128)
            S = torch.zeros(R, 1, 1, 128)
            nbv = N - 32
            nv = torch.tensor([nbv - 40 + (i % n) + 7 * (i // n) for i in range(R)], dtype=torch.float32)
            st = {
                "C": up(C, ttnn.bfloat16, ttnn.TILE_LAYOUT),
                "S": up(S, ttnn.bfloat16, ttnn.TILE_LAYOUT),
                "nvalid": up(nv.reshape(R, 1, 1, 1), ttnn.float32, ttnn.TILE_LAYOUT),
                "valid": up(
                    torch.full((1, 1, 1, 1), int(nv.max()), dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
                ),
            }
            res = {}
            for fused in (False, True):
                SP._FUSED_SCORE = fused
                f = lambda: idx.forward(x, qr, st)
                res[fused] = ttnn.to_torch(f()).reshape(R, -1)
                res[(fused, "us")] = traced_us(md, lambda: idx.forward(x, qr, {k: v for k, v in st.items()}))
            eq = torch.equal(res[False], res[True])
            same_set = all(set(res[False][r].tolist()) == set(res[True][r].tolist()) for r in range(R))
            print(
                f"FWD U={U} n={n} N={N} tie={tie}: ids identical {eq} (sets equal {same_set})  chain {res[(False,'us')]:.1f} us  fused {res[(True,'us')]:.1f} us",
                flush=True,
            )
            assert eq
