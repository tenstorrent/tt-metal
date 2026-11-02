# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A3 step (b): ternary matmul with per-128-group fp16 scales applied in the kernel.

Reference is float64 x @ (codes * scales), i.e. cpu_ref's (T*s) @ x without the
Hadamard (that is applied to x upstream and is not part of this op).

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.a3_scaled [--k 5120 --n 17408] [--wall 100]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from .a1_bfp2_shard import all_kernel_us
from .a1_calib import device_kernel_us
from .a3_tmm import assemble


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=17408)
    ap.add_argument("--n", type=int, default=5120)
    ap.add_argument("--fmt", default="bfp2")
    ap.add_argument("--wpb", default="4")
    ap.add_argument("--calls", type=int, default=9)
    ap.add_argument("--wall", type=int, default=0)
    args = ap.parse_args()
    if not args.wall:
        for key, val in (("TT_METAL_DEVICE_PROFILER", "1"), ("TT_METAL_PROFILER_CPP_POST_PROCESS", "1"),
                         ("TT_METAL_PROFILER_MID_RUN_DUMP", "1")):
            os.environ.setdefault(key, val)

    import ttnn
    from models.demos.blackhole.bonsai2.native.a3 import op as a3

    k, n = args.k, args.n
    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    dtypes = {"bfp2": ttnn.DataType.BFLOAT2_B, "bfp4": ttnn.bfloat4_b}
    rc = 0
    try:
        banks = device.dram_grid_size().x
        rng = np.random.default_rng(0)
        x = torch.randn(k).bfloat16().float()
        codes = torch.from_numpy(rng.integers(-1, 2, (k, n)).astype(np.float32))
        scales = torch.from_numpy((rng.random((k // 128, n)) * 0.03 + 0.005).astype(np.float16).astype(np.float32))
        wd = codes.double() * scales.double().repeat_interleave(128, 0)
        ref = (x.double() @ wd).numpy()
        w = ttnn.to_device(ttnn.from_torch(a3.shuffle_for_banks(codes.reshape(1, 1, k, n), banks),
                                           dtype=dtypes[args.fmt], layout=ttnn.TILE_LAYOUT),
                           device, memory_config=a3.weight_mem(device, k, n))
        srows = a3.scale_rows(scales)
        s = ttnn.to_device(ttnn.from_torch(a3.shuffle_for_banks(srows.reshape(1, 1, *srows.shape), banks),
                                           dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT),
                           device, memory_config=a3.weight_mem(device, srows.shape[0], n))
        x8 = a3.mask_x(x)
        mb_codes = (n * k // 4 + n * k // 16) / 1e6 if args.fmt == "bfp2" else (n * k // 2 + n * k // 16) / 1e6
        mb = mb_codes + srows.numel() * 4 / 1e6
        print(f"a3 scaled K={k} N={n} {args.fmt}: codes {mb_codes:.2f} MB + scales {srows.numel() * 4 / 1e6:.2f} MB")
        for wpb in [int(v) for v in args.wpb.split(",")]:
            workers = a3.worker_cores(device, wpb)
            cores = [c for c, _, _ in workers]
            pcn = n // 32 // len(workers)
            x8t = ttnn.from_torch(x8.repeat(len(workers), 1).reshape(1, 1, 8 * len(workers), k), dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile([8, 32]), device=device,
                                  memory_config=ttnn.MemoryConfig(
                                      ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                                      ttnn.ShardSpec(a3.core_set(cores), [8, k], ttnn.ShardOrientation.ROW_MAJOR)))
            yt = ttnn.from_torch(torch.full((1, 1, 1, n), -7.0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                 tile=ttnn.Tile([1, 32]), device=device, memory_config=a3.out_mem(cores, n))
            try:
                a3.ternary_mm_scaled(x8t, w, s, yt, workers)
                ttnn.synchronize_device(device)
                y = assemble(ttnn.to_torch(yt).double().flatten().numpy(), workers, pcn)
                pcc = float(np.corrcoef(y, ref)[0, 1])
                rel = float(np.abs(y - ref).max() / np.abs(ref).max())
                np.savez(os.path.expanduser("~/avan/native2bit/agentA/tmp/a3s.npz"), y=y, ref=ref)
                if args.wall:
                    t0 = time.perf_counter()
                    for _ in range(args.wall):
                        a3.ternary_mm_scaled(x8t, w, s, yt, workers)
                    ttnn.synchronize_device(device)
                    devs = [(time.perf_counter() - t0) / args.wall * 1e6]
                else:
                    device_kernel_us(ttnn, device)
                    for _ in range(args.calls):
                        a3.ternary_mm_scaled(x8t, w, s, yt, workers)
                    ttnn.synchronize_device(device)
                    devs = all_kernel_us(ttnn, device)
                dev = float(np.median(devs))
                print(f"{args.fmt} wpb={wpb} {dev:7.1f} us {'wall' if args.wall else 'dev'} {mb * 1e3 / dev:6.1f} GB/s "
                      f"pcc {pcc:.7f} maxrel {rel:.2e} n={len(devs)} [{min(devs):.1f}-{max(devs):.1f}]", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"{args.fmt} wpb={wpb} FAILED {type(e).__name__}: {str(e)[:400]}", flush=True)
                rc = 1
    finally:
        ttnn.close_device(device)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
