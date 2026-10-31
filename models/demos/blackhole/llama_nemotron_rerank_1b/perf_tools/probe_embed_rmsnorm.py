# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""embed_rmsnorm (gather into the resident residual + first RMSNorm) vs ttnn.embedding + to_resident +
resident_rmsnorm: exactness of the residual, y PCC, traced time.

    python models/demos/blackhole/llama_nemotron_rerank_1b/perf_tools/probe_embed_rmsnorm.py [--batch 32 --seq 512]
"""
import argparse
import time

import torch

import ttnn
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.custom_ops.matmul_add_norm import (
    embed_rmsnorm,
    from_resident,
    make_matmul_add_norm_constants,
    resident_rmsnorm,
    to_resident,
)

W, V, EPS, REPS = 2048, 128256, 1e-5, 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--defines", default="", help="comma-separated kernel defines for embed_rmsnorm")
    args = ap.parse_args()
    B, S = args.batch, args.seq
    M = B * S
    defines = [tuple(d.split("=", 1)) if "=" in d else (d, "1") for d in args.defines.split(",") if d]
    dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=50_000_000)
    grid = dev.compute_with_storage_grid_size()
    torch.manual_seed(0)
    table_h = torch.randn(V, W)
    table_h[:, 7] *= 40
    table = ttnn.from_torch(table_h, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    ids = torch.randint(0, V, (B, S), dtype=torch.int32)
    tok = ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    consts = make_matmul_add_norm_constants(torch.ones(W), dev)

    def new():
        return embed_rmsnorm(tok, table, consts, rows=M, width=W, eps=EPS, grid=grid, skip_gamma=True, defines=defines)

    def old():
        x = ttnn.embedding(tok, table, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        res = to_resident(ttnn.reshape(x, [1, 1, M, W]), grid)
        ttnn.deallocate(x)
        return res, resident_rmsnorm(res, consts, rows=M, width=W, eps=EPS, grid=grid, skip_gamma=True)

    ref_x = table_h.to(torch.bfloat16).float()[ids.reshape(-1).long()]
    ref_y = ref_x * torch.rsqrt(ref_x.pow(2).mean(-1, keepdim=True) + EPS)
    for name, fn in (("embed_rmsnorm", new), ("old", old)):
        res, y = fn()
        r = ttnn.to_torch(from_resident(res, M, W)).float().reshape(M, W)
        yy = ttnn.to_torch(y).float().reshape(M, W)
        p = torch.corrcoef(torch.stack([yy.flatten().double(), ref_y.flatten().double()]))[0, 1].item()
        print(
            f"{name}: residual max|diff| {(r - ref_x).abs().max():.4g} | y pcc {p:.6f} max|err| {(yy - ref_y).abs().max():.4f}"
        )
        bad = ((r - ref_x).abs() > 0).reshape(M // 32, 32, W // 32, 32).any(3)  # [row tile, row, col tile]
        if bad.any():
            bt = bad.any(1)
            print(
                f"  bad tiles {int(bt.sum())}/{bt.numel()}; bad col tiles {bt.any(0).nonzero().flatten().tolist()[:20]}"
            )
            print(
                f"  bad row tiles (first 20) {bt.any(1).nonzero().flatten().tolist()[:20]}; bad rows-in-tile {bad.any(2).any(0).nonzero().flatten().tolist()}"
            )
        ttnn.deallocate(res)
        ttnn.deallocate(y)

    def run_free(fn):
        # Each call allocates an L1 residual: free it right away so REPS of them fit.
        for t in fn():
            ttnn.deallocate(t)

    def timed(fn):
        run_free(fn)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        for _ in range(REPS):
            run_free(fn)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.synchronize_device(dev)
        best = 1e9
        for _ in range(3):
            t = time.perf_counter()
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(dev)
            best = min(best, (time.perf_counter() - t) / REPS * 1e6)
        ttnn.release_trace(dev, tid)
        return best

    print(
        f"B={B} S={S}: embed_rmsnorm {timed(new):.1f} us, embedding + to_resident + resident_rmsnorm {timed(old):.1f} us"
    )
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
