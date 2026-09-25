# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Sweep matmul program configs for Chronos folded-batch linears on one P150."""

import itertools
import math
import sys
import time

import torch
import ttnn

B, T, TP = 1024, 133, 160
SHAPES = {
    "qkv": (768, 2304),
    "wo": (768, 768),
    "ff_up": (768, 3072),
    "ff_down": (3072, 768),
}
L1_BUDGET = 1_200_000


def ckc(fid=ttnn.MathFidelity.HiFi2):
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )


def bench(fn, iters=4):
    out = fn()
    ttnn.synchronize_device(dev)
    ttnn.deallocate(out)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
        ttnn.deallocate(out)
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / iters * 1000


def subblocks(bh, bw):
    best = None
    for h, w in itertools.product(range(1, 9), range(1, 9)):
        if h * w > 8 or bh % h or bw % w:
            continue
        if best is None or h * w > best[0] * best[1] or (h * w == best[0] * best[1] and w > best[1]):
            best = (h, w)
    return best


def configs_2d(Mt, Kt, Nt, act):
    for gx, gy in [(11, 10), (8, 10)]:
        pcm = math.ceil(Mt / gy)
        pcn = math.ceil(Nt / gx)
        if (math.ceil(Nt / pcn)) < gx - 1:
            continue
        for obh in [d for d in range(1, pcm + 1) if pcm % d == 0 and 8 <= d <= 32]:
            for obw in [pcn]:
                for ibw in [d for d in range(4, Kt + 1) if Kt % d == 0 and d <= 32]:
                    tiles = 2 * obh * ibw + 2 * ibw * obw + obh * obw
                    if tiles * 2048 > L1_BUDGET or obh < 4:
                        continue
                    sh, sw = subblocks(obh, obw)
                    yield (
                        f"2d g{gx}x{gy} pcm{pcm} pcn{pcn} obh{obh} ibw{ibw} sb{sh}x{sw}",
                        ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                            compute_with_storage_grid_size=(gx, gy),
                            in0_block_w=ibw,
                            out_subblock_h=sh,
                            out_subblock_w=sw,
                            out_block_h=obh,
                            out_block_w=obw,
                            per_core_M=pcm,
                            per_core_N=pcn,
                            transpose_mcast=False,
                            fused_activation=act,
                            fuse_batch=True,
                        ),
                    )


def main(which, max_cfgs=400):
    K, N = SHAPES[which]
    Mt, Kt, Nt = B * TP // 32, K // 32, N // 32
    x = ttnn.from_torch(
        torch.randn(B, T, K),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    w = ttnn.from_torch(
        torch.randn(K, N) / K**0.5,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    act = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU) if which == "ff_up" else None
    base = bench(lambda: ttnn.linear(x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ckc()))
    print(f"[{which}] auto: {base:.2f} ms", flush=True)
    results = []
    for i, (name, cfg) in enumerate(configs_2d(Mt, Kt, Nt, act)):
        if i >= max_cfgs:
            break
        try:
            ms = bench(
                lambda: ttnn.linear(
                    x, w, program_config=cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=ckc()
                )
            )
        except Exception as e:  # noqa: BLE001
            print(f"[{which}] {name}: ERR {str(e).splitlines()[0][:160]}", flush=True)
            continue
        results.append((ms, name))
        print(f"[{which}] {name}: {ms:.2f} ms", flush=True)
    results.sort()
    print(f"\n[{which}] best:")
    for ms, name in results[:8]:
        print(f"   {ms:.2f} ms  {name}")


if __name__ == "__main__":
    dev = ttnn.open_device(device_id=0, l1_small_size=0)
    dev.enable_program_cache()
    try:
        for which in sys.argv[1:]:
            main(which)
    finally:
        ttnn.close_device(dev)
