# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""matmul_qkv_rope vs minimal_matmul + fused_heads_rope: accuracy against torch and traced time.

    python models/demos/blackhole/llama_nemotron_rerank_1b/perf_tools/probe_qkv_rope.py --batch 32 --seq 512
"""

import argparse
import time

import torch

import ttnn
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.custom_ops.fused_heads_rope import fused_heads_rope
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.custom_ops.matmul_qkv_rope import (
    make_rope_table,
    matmul_qkv_rope,
)
from models.tt_transformers.tt.common import get_rot_transformation_mat, precompute_freqs

H, KV, D, DIM = 32, 8, 64, 2048
REPS = 10


def timed(device, fn):
    fn()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    outs = [fn() for _ in range(REPS)]
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    best = 1e9
    for _ in range(3):
        t = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        best = min(best, (time.perf_counter() - t) / REPS * 1e6)
    ttnn.release_trace(device, tid)
    for o in outs:
        for t in o:
            ttnn.deallocate(t)
    return best


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def rotate_half(x):
    return torch.cat([-x[..., D // 2 :], x[..., : D // 2]], dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--block", default="8,8,4,1,4", help="mb,kb,nb,sh,sw")
    ap.add_argument("--xdtype", default="bfloat8_b")
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--rope-fidelity", default=None)
    ap.add_argument(
        "--bf16-acc", action="store_true", help="bf16 DST (8-tile subblocks); the L1 accumulator stays fp32"
    )
    ap.add_argument("--in0-mcast", action="store_true")
    ap.add_argument("--defines", default="", help="comma-separated kernel defines (NAME or NAME=VAL)")
    args = ap.parse_args()
    block = tuple(int(v) for v in args.block.split(","))
    defines = [tuple(d.split("=", 1)) if "=" in d else (d, "1") for d in args.defines.split(",") if d]
    B, S = args.batch, args.seq
    M = B * S

    device = ttnn.open_device(device_id=0, l1_small_size=0, trace_region_size=200_000_000)
    torch.manual_seed(0)
    x = torch.randn(1, 1, M, DIM)
    w = torch.randn(DIM, (H + 2 * KV) * D) / DIM**0.5
    cos, sin = precompute_freqs(D, 2 * 8192, 500000.0, 32.0, 8192)
    cos, sin = cos[:S].float(), sin[:S].float()

    tx = ttnn.from_torch(x, dtype=getattr(ttnn, args.xdtype), layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    table = make_rope_table(cos, sin, device)

    def new():
        return matmul_qkv_rope(
            tx,
            tw,
            table,
            batch=B,
            n_heads=H,
            n_kv_heads=KV,
            block=block,
            rope_fidelity=args.rope_fidelity,
            defines=defines,
            fp32_acc=not args.bf16_acc,
            in0_mcast=args.in0_mcast,
        )

    q, k, v = new()
    xr = ttnn.to_torch(tx).float()
    wr = ttnn.to_torch(tw).float()
    qkv = (xr @ wr).reshape(B, S, H + 2 * KV, D).transpose(1, 2)
    cf, sf = torch.cat([cos, cos], -1), torch.cat([sin, sin], -1)
    rq = qkv[:, :H] * cf + rotate_half(qkv[:, :H]) * sf
    rk = qkv[:, H : H + KV] * cf + rotate_half(qkv[:, H : H + KV]) * sf
    rv = qkv[:, H + KV :]
    for name, got, ref in (("q", q, rq), ("k", k, rk), ("v", v, rv)):
        g = ttnn.to_torch(got).float()
        print(f"{name}: shape {tuple(g.shape)} pcc {pcc(g, ref):.6f} max|err| {(g - ref).abs().max():.4f}")
        ttnn.deallocate(got)

    if not args.no_time:
        trans = ttnn.from_torch(
            get_rot_transformation_mat(D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        cos_l = ttnn.from_torch(torch.randn(1, 1, S, D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        sin_l = ttnn.from_torch(torch.randn(1, 1, S, D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        ckc = ttnn.init_device_compute_kernel_config(
            ttnn.device.Arch.BLACKHOLE,
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        def old():
            qkv = ttnn.experimental.minimal_matmul(
                tx,
                tw,
                config=ttnn.MinimalMatmulConfig(
                    M_block_size=8,
                    K_block_size=32,
                    N_block_size=4,
                    subblock_h=1,
                    subblock_w=4,
                    compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                ),
                compute_kernel_config=ckc,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            out = fused_heads_rope(qkv, cos_l, sin_l, trans, n_heads=H, n_kv_heads=KV, batch=B)
            ttnn.deallocate(qkv)
            return out

        print(f"B={B} S={S} block={block}: fused {timed(device, new):.1f} us, old {timed(device, old):.1f} us")
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
