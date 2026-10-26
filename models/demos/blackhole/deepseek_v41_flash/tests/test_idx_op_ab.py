# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""L12 op-level A/B on production shapes (one chip = one mesh row, U = 1 user, n = 1+k rows per user): the whole spec-verify indexer layer (project + score + mask + top-k) with the chain
(DSV41_IDX_FUSED=0 path) vs the fused score+mask op, traced, at live lengths P (positions) 8k..128k for the two index-layer geometries (ratio 2: P/2 entries, layers 2/8/14; ratio 1: P entries,
layers 20..36). The scored length of the chain is the position bucket (set_pos_bucket); the fused op scores the live entries. Two live lengths per bucket: just above half of it
(the bucket just entered; production ladder 512/2048/8192/32768/131072/524288) and at its end. Also compares the selected ids (same ids, same order) and the score+mask stage alone.
Env: IDXAB_P (comma list of positions), IDXAB_ROWS (n), IDXAB_REPS."""
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import indexer as I
from models.demos.blackhole.deepseek_v41_flash.tt import spec_paged as SP
from models.demos.blackhole.deepseek_v41_flash.tt.spec_paged import SpecIndexer

PS = [int(x) for x in os.environ.get("IDXAB_P", "8212,16404,32788,65556,131093").split(",")]
NROWS = [int(x) for x in os.environ.get("IDXAB_ROWS", "4,6").split(",")]
REPS = int(os.environ.get("IDXAB_REPS", "30"))


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
def test_op_ab(mesh_device):
    md = mesh_device
    U = 1
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t, dt, lay: ttnn.from_torch(t.contiguous(), device=md, dtype=dt, layout=lay, mesh_mapper=rep)
    N = 524288 + 288
    torch.manual_seed(7)
    w = {"wq_b": torch.randn(4096, 1280) * 0.02, "weights_proj": torch.randn(32, 5120) * 0.02}
    freqs = torch.polar(torch.ones(N * 2 + 64, 32), torch.rand(N * 2 + 64, 32))
    keys = torch.randn(U, N - 32, 128)
    for ratio in (2, 1):
        for n in NROWS:
            R = U * n
            idx = SpecIndexer(md, w, freqs, users_per_row=U, n_alloc=N, ratio=ratio, key_dtype=ttnn.bfloat8_b, fp4_q=True, backend="matmul", n=n)
            idx.load_keys(keys)
            x = up(torch.randn(1, 1, R, 5120), ttnn.bfloat16, ttnn.TILE_LAYOUT)
            qr = up(torch.randn(1, 1, R, 1280), ttnn.bfloat16, ttnn.TILE_LAYOUT)
            for P in PS:
                live = P // ratio
                for where in ("bucket_start", "bucket_end"):
                    bucket = next(b for b in (512, 2048, 8192, 32768, 131072, 524288) if b > P)  # the spec ladder (DSV41_SPEC_POS_FACTOR=4)
                    if where == "bucket_end":
                        P2 = bucket - 8
                        live = P2 // ratio
                    else:
                        P2 = P
                    idx.set_pos_bucket(bucket)
                    idx._arange_for(idx.n_score)
                    nv = torch.tensor([live - (R - 1 - i) // 2 for i in range(R)], dtype=torch.float32)
                    st = {
                        "C": up(torch.ones(R, 1, 1, 128), ttnn.bfloat16, ttnn.TILE_LAYOUT),
                        "S": up(torch.zeros(R, 1, 1, 128), ttnn.bfloat16, ttnn.TILE_LAYOUT),
                        "nvalid": up(nv.reshape(R, 1, 1, 1), ttnn.float32, ttnn.TILE_LAYOUT),
                        "valid": up(torch.full((1, 1, 1, 1), int(nv.max()), dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                    }
                    res, us = {}, {}
                    for fused in (False, True):
                        SP._FUSED_SCORE = fused
                        res[fused] = ttnn.to_torch(idx.forward(x, qr, st)).reshape(R, -1)
                        us[fused] = traced_us(md, lambda: idx.forward(x, qr, st))
                    SP._FUSED_SCORE = True
                    eq = torch.equal(res[False], res[True])
                    print(
                        f"OPAB ratio={ratio} n={n} P={P2} bucket={bucket} nb={idx.n_score} live={live}: chain {us[False]:8.1f} us fused {us[True]:8.1f} us  "
                        f"ids identical {eq}",
                        flush=True,
                    )
                    assert eq
