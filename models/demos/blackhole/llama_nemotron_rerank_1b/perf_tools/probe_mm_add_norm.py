# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""matmul_add_norm vs minimal_matmul + fused_add_rmsnorm: accuracy against torch and traced time.

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
    make_matmul_add_norm_constants,
    matmul_add_norm,
)

SHAPES = {
    "wo": (2048, 2048, "bfloat16", "HiFi2", (8, 8, 8, 1, 4)),
    "ff2": (8192, 2048, "bfloat8_b", "LoFi", (8, 16, 8, 1, 4)),
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
        for t in o:
            ttnn.deallocate(t)
    return best


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="wo", choices=list(SHAPES))
    ap.add_argument("--m", type=int, default=16384)
    ap.add_argument("--block", default=None, help="mb,kb,nb,sh,sw")
    ap.add_argument("--grid", default=None, help="gx,gy")
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--no-time-old", action="store_true")
    ap.add_argument("--xdtype", default=None)
    ap.add_argument("--in1-depth", type=int, default=2)
    ap.add_argument("--in0-depth", type=int, default=2)
    ap.add_argument("--y-l1", action="store_true")
    ap.add_argument("--debug-gather", action="store_true")
    ap.add_argument("--defines", default="", help="comma-separated compute defines for the fused op (timing variants)")
    args = ap.parse_args()
    K, N, xdt, fid, block = SHAPES[args.kind]
    if args.block:
        block = tuple(int(v) for v in args.block.split(","))
    xdt = getattr(ttnn, args.xdtype or xdt)
    fid = getattr(ttnn.MathFidelity, fid)

    device = ttnn.open_device(device_id=0, l1_small_size=0, trace_region_size=200_000_000)
    torch.manual_seed(0)
    x = torch.randn(1, 1, args.m, K) * 0.5
    w = torch.randn(K, N) / K**0.5
    r = torch.randn(1, 1, args.m, N) * 2
    r[..., 7] *= 40  # an outlier channel, as in Llama's residual stream
    gamma = torch.rand(N) + 0.5

    tx = ttnn.from_torch(x, dtype=xdt, layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    tr = ttnn.from_torch(r, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    if "SPLIT_RESIDUAL" in args.defines:
        # column-split layout: [gy, M, nb * 32], core row yy's N slice of every row contiguous
        gy, nb = (int(args.grid.split(",")[1]) if args.grid else 10), block[2]
        rp = torch.zeros(gy, args.m, nb * 32)
        for yy in range(gy):
            c0 = yy * (-(-N // 32 // gy)) * 32
            w_ = min(N - c0, nb * 32)
            if w_ > 0:
                rp[yy, :, :w_] = r[0, 0, :, c0 : c0 + w_]
        tr = ttnn.from_torch(
            rp.reshape(1, 1, gy * args.m, nb * 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
    consts = make_matmul_add_norm_constants(gamma, device)
    old_consts = make_add_norm_constants(gamma, EPS, device)
    ckc = ttnn.init_device_compute_kernel_config(
        ttnn.device.Arch.BLACKHOLE, math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    grid = None
    if args.grid:
        gx, gy = (int(v) for v in args.grid.split(","))
        grid = ttnn.CoreCoord(gx, gy)

    def new():
        return matmul_add_norm(
            tx,
            tw,
            tr,
            consts,
            eps=EPS,
            block=block,
            grid=grid,
            fidelity=fid,
            defines=[(d, "1") for d in args.defines.split(",") if d],
        )

    def old():
        mm = ttnn.experimental.minimal_matmul(
            tx,
            tw,
            config=ttnn.MinimalMatmulConfig(
                M_block_size=block[0],
                K_block_size=block[1],
                N_block_size=block[2],
                subblock_h=block[3],
                subblock_w=block[4],
                compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            ),
            compute_kernel_config=ckc,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        h, y = fused_add_rmsnorm(tr, mm, old_consts, out_dtype=ttnn.bfloat8_b, sum_dtype=ttnn.bfloat16)
        ttnn.deallocate(mm)
        return h, y

    if args.debug_gather:
        h, y = matmul_add_norm(
            tx,
            tw,
            tr,
            consts,
            eps=EPS,
            block=block,
            grid=grid,
            fidelity=fid,
            y_dtype=ttnn.float32,
            defines=[("DEBUG_GATHER", "1")],
        )
        ty = ttnn.to_torch(y).float()[0, 0]
        ref_h = (r + x @ w)[0, 0]
        gyy = grid.y if grid else 10
        npc = -(-N // 32 // gyy)
        for rt in (0, 1, 5):
            G = ty[rt * 32 : (rt + 1) * 32, 0:32]
            inv = ty[rt * 32 : (rt + 1) * 32, 32:64]
            rows = ref_h[rt * 32 : (rt + 1) * 32]
            exp_part = torch.stack(
                [rows[:, p * npc * 32 : (p + 1) * npc * 32].pow(2).sum(-1) / N for p in range(gyy)], -1
            )
            print(f"row tile {rt}: G[0,:12]", [round(v, 4) for v in G[0, :12].tolist()])
            print("   expected partials[0]", [round(v, 4) for v in exp_part[0].tolist()])
            print(
                "   G[17,:12]",
                [round(v, 4) for v in G[17, :12].tolist()],
                "exp",
                [round(v, 4) for v in exp_part[17].tolist()],
            )
            print(
                "   inv[0,:3]",
                inv[0, :3].tolist(),
                "inv[17,0]",
                inv[17, 0].item(),
                "true",
                torch.rsqrt(rows.pow(2).mean(-1) + EPS)[[0, 17]].tolist(),
            )
        return
    ref_h = r + x @ w
    if "SPLIT_RESIDUAL" in args.defines:
        print("(split layout: h accuracy not checked)")
    ref_y = ref_h * torch.rsqrt(ref_h.pow(2).mean(-1, keepdim=True) + EPS) * gamma
    split = "SPLIT_RESIDUAL" in args.defines
    for name, fn in (("fused", new),) + ((("old", old),) if not split else ()):
        h, y = fn()
        ty = ttnn.to_torch(y).float()
        th = ref_h if split else ttnn.to_torch(h).float()
        print(
            f"{name}: h pcc {pcc(th, ref_h):.6f} max|err| {(th - ref_h).abs().max():.4f} | "
            f"y pcc {pcc(ty, ref_y):.6f} max|err| {(ty - ref_y).abs().max():.4f}"
        )
        if name == "fused":
            e = (th - ref_h).abs().reshape(args.m // 32, 32, N // 32, 32).amax(dim=(1, 3))
            bad = (e > 1.0).nonzero().tolist()
            print("bad h tiles", len(bad), "of", e.numel(), "cols", sorted({c for _, c in bad})[:80])
            print("rows", sorted({r for r, _ in bad})[:80])
            hg = (ref_h * gamma)[0, 0]
            ratio = ty[0, 0] / hg
            true_inv = torch.rsqrt(ref_h.pow(2).mean(-1) + EPS)[0, 0]
            print("nan y", torch.isnan(ty).sum().item(), "inf", torch.isinf(ty).sum().item())
            for row in (0, 1, 31, 32, 100, 500):
                rr = ratio[row]
                print(
                    f"row {row}: true inv {true_inv[row]:.5f} got median {rr.median():.5f} per-coltile medians",
                    [round(v, 4) for v in rr.reshape(-1, 32).median(-1).values.tolist()[:12]],
                )
            mm = x @ w
            print(
                "tile(0,0) h vs ref",
                th[0, 0, 0, :4].tolist(),
                ref_h[0, 0, 0, :4].tolist(),
                "mm",
                mm[0, 0, 0, :4].tolist(),
                "r",
                r[0, 0, 0, :4].tolist(),
            )
        ttnn.deallocate(h)
        ttnn.deallocate(y)
    if not args.no_time:
        t_old = 0.0 if args.no_time_old else timed(device, old)
        print(f"{args.kind} M={args.m} block={block}: fused {timed(device, new):.1f} us, old {t_old:.1f} us")
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
