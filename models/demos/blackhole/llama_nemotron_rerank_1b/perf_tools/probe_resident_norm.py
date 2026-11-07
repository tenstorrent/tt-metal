# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""resident_rmsnorm (on the L1-resident residual) vs ttnn.rms_norm on the interleaved tensor: accuracy, time."""
import time

import torch

import ttnn
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.custom_ops.matmul_add_norm import (
    make_matmul_add_norm_constants,
    resident_rmsnorm,
    to_resident,
)

M, W, EPS, REPS = 16384, 2048, 1e-5, 10
dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=50_000_000)
grid = dev.compute_with_storage_grid_size()
torch.manual_seed(0)
x = torch.randn(1, 1, M, W) * 2
x[..., 7] *= 40
tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
res = to_resident(tx, grid)
consts = make_matmul_add_norm_constants(torch.ones(W), dev)
ones_w = ttnn.from_torch(torch.ones(1, 1, W // 32, 32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)


def new():
    return resident_rmsnorm(res, consts, rows=M, width=W, eps=EPS, grid=grid, skip_gamma=True)


def old():
    return ttnn.rms_norm(tx, epsilon=EPS, weight=ones_w)


def timed(fn):
    fn()
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    outs = [fn() for _ in range(REPS)]
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.synchronize_device(dev)
    best = 1e9
    for _ in range(3):
        t = time.perf_counter()
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        best = min(best, (time.perf_counter() - t) / REPS * 1e6)
    ttnn.release_trace(dev, tid)
    for o in outs:
        ttnn.deallocate(o)
    return best


y = ttnn.to_torch(new()).float()
d = torch.corrcoef(torch.stack([y.flatten().double(), ref.flatten().double()]))[0, 1].item()
print(f"resident_rmsnorm pcc {d:.6f} max|err| {(y - ref).abs().max():.4f}")
print(f"resident_rmsnorm {timed(new):.1f} us, ttnn.rms_norm {timed(old):.1f} us")
ttnn.close_device(dev)
