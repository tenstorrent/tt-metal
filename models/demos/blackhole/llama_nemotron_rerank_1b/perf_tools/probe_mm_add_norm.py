# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""matmul_add_norm (L1-resident residual) vs minimal_matmul + fused_add_rmsnorm: accuracy against
torch and traced time.

    python models/demos/blackhole/llama_nemotron_rerank_1b/perf_tools/probe_mm_add_norm.py --kind wo --m 16384
"""

import argparse
import time

import torch

import ttnn
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.custom_ops.fused_add_rmsnorm import (
    fused_add_rmsnorm,
    make_add_norm_constants,
)
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.custom_ops.matmul_add_norm import (
    from_resident,
    make_matmul_add_norm_constants,
    matmul_add_norm,
    to_resident,
)

# kind: (K, N, x dtype, fidelity, block for the fused op, block for minimal_matmul)
SHAPES = {
    "wo": (2048, 2048, "bfloat16", "HiFi2", (4, 8, 8, 1, 4), (8, 8, 4, 1, 4)),
    "ff2": (8192, 2048, "bfloat8_b", "LoFi", (4, 8, 8, 1, 4), (8, 16, 8, 1, 4)),
}
EPS = 1e-5
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
        for t in o if isinstance(o, tuple) else (o,):
            ttnn.deallocate(t)
    return best


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="wo", choices=list(SHAPES))
    ap.add_argument("--m", type=int, default=16384)
    ap.add_argument("--block", default=None, help="mb,kb,nb,sh,sw for the fused op")
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--no-time-old", action="store_true")
    ap.add_argument("--bf16-acc", action="store_true")
    ap.add_argument("--in0-depth", type=int, default=2)
    ap.add_argument("--full-sync", action="store_true")
    ap.add_argument("--acc-cb-fp32", action="store_true")
    ap.add_argument("--flat", action="store_true", help="non-transposed grid: M over y, N over x")
    ap.add_argument("--in1-depth", type=int, default=2)
    ap.add_argument("--in0-mcast", action="store_true")
    ap.add_argument("--defines", default="", help="comma-separated kernel defines for the fused op")
    args = ap.parse_args()
    K, N, xdt, fid, block, old_block = SHAPES[args.kind]
    if args.block:
        block = tuple(int(v) for v in args.block.split(","))
    xdt, fid = getattr(ttnn, xdt), getattr(ttnn.MathFidelity, fid)
    defines = [tuple(d.split("=", 1)) if "=" in d else (d, "1") for d in args.defines.split(",") if d]

    device = ttnn.open_device(device_id=0, l1_small_size=0, trace_region_size=200_000_000)
    grid = device.compute_with_storage_grid_size()
    torch.manual_seed(0)
    x = torch.randn(1, 1, args.m, K) * 0.5
    w = torch.randn(K, N) / K**0.5
    r = torch.randn(1, 1, args.m, N) * 2
    r[..., 7] *= 40  # an outlier channel, as in Llama's residual stream
    gamma = torch.rand(N) + 0.5

    tx = ttnn.from_torch(x, dtype=xdt, layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    tr = ttnn.from_torch(r, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    res = to_resident(tr, grid, transposed=not args.flat)
    consts = make_matmul_add_norm_constants(gamma, device)
    old_consts = make_add_norm_constants(gamma, EPS, device)
    ckc = ttnn.init_device_compute_kernel_config(
        ttnn.device.Arch.BLACKHOLE, math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )

    def new():
        return matmul_add_norm(
            tx,
            tw,
            res,
            consts,
            eps=EPS,
            block=block,
            fidelity=fid,
            defines=defines,
            fp32_acc=not args.bf16_acc,
            in0_depth=args.in0_depth,
            in1_depth=args.in1_depth,
            full_sync=args.full_sync,
            transposed=not args.flat,
            acc_cb_fp32=True if args.acc_cb_fp32 else None,
            in0_mcast=args.in0_mcast,
        )

    def old():
        mm = ttnn.experimental.minimal_matmul(
            tx,
            tw,
            config=ttnn.MinimalMatmulConfig(
                M_block_size=old_block[0],
                K_block_size=old_block[1],
                N_block_size=old_block[2],
                subblock_h=old_block[3],
                subblock_w=old_block[4],
                compute_with_storage_grid_size=grid,
            ),
            compute_kernel_config=ckc,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        h, y = fused_add_rmsnorm(tr, mm, old_consts, out_dtype=ttnn.bfloat8_b, sum_dtype=ttnn.bfloat16)
        ttnn.deallocate(mm)
        return h, y

    ref_h = r + ttnn.to_torch(tx).float() @ ttnn.to_torch(tw).float()
    ref_y = ref_h * torch.rsqrt(ref_h.pow(2).mean(-1, keepdim=True) + EPS) * gamma
    y = new()
    th = ttnn.to_torch(from_resident(res, args.m, N)).float()
    ty = ttnn.to_torch(y).float()
    print(
        f"fused: h pcc {pcc(th, ref_h):.6f} max|err| {(th - ref_h).abs().max():.4f} | "
        f"y pcc {pcc(ty, ref_y):.6f} max|err| {(ty - ref_y).abs().max():.4f}"
    )
    if not args.no_time_old:
        h_old, y_old = old()
        print(
            f"old:   h pcc {pcc(ttnn.to_torch(h_old).float(), ref_h):.6f} | y pcc {pcc(ttnn.to_torch(y_old).float(), ref_y):.6f}"
        )
    if not args.no_time:
        t_old = 0.0 if args.no_time_old else timed(device, old)
        print(f"{args.kind} M={args.m} block={block}: fused {timed(device, new):.1f} us, old {t_old:.1f} us")
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
