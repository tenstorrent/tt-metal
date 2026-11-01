# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A3 ternary matmul (native/a3/op.py): correctness and device time.

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.a3_tmm [--k 5120 --n 17408] [--xint]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .a1_bfp2_shard import all_kernel_us, weight_mb
from .a1_calib import device_kernel_us


def assemble(y, workers, pcn: int):
    """Device output (shard i = workers[i], measured) -> [N] in column order."""
    wpb = len(workers) // len({b for _, b, _ in workers})
    parts = {b * wpb + s: y[i * pcn * 32:(i + 1) * pcn * 32] for i, (_, b, s) in enumerate(workers)}
    return np.concatenate([parts[j] for j in sorted(parts)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=17408)
    ap.add_argument("--n", type=int, default=5120)
    ap.add_argument("--fmts", default="bfp2,bfp4")
    ap.add_argument("--wpb", default="1,2")
    ap.add_argument("--subblock-k", default="34,68")
    ap.add_argument("--calls", type=int, default=9)
    ap.add_argument("--xint", action="store_true", help="x = small integers, so the exact answer is known")
    args = ap.parse_args()
    for key, val in (("TT_METAL_DEVICE_PROFILER", "1"), ("TT_METAL_PROFILER_CPP_POST_PROCESS", "1"),
                     ("TT_METAL_PROFILER_MID_RUN_DUMP", "1")):
        os.environ.setdefault(key, val)

    import ttnn
    from models.demos.blackhole.bonsai2.native.a3 import op as a3

    k, n = args.k, args.n
    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    dtypes = {"bfp2": ttnn.DataType.BFLOAT2_B, "bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b}
    tile = ttnn.Tile([1, 32])
    rc = 0
    try:
        banks = device.dram_grid_size().x
        x = (torch.randint(-7, 8, (1, k)).float() if args.xint else torch.randn(1, k).bfloat16().float())
        codes = torch.from_numpy(np.random.default_rng(0).integers(-1, 2, (k, n)).astype(np.float32))
        ref = (x.double() @ codes.double()).flatten().numpy()
        shuffled = a3.shuffle_for_banks(codes.reshape(1, 1, k, n), banks)
        print(f"a3 ternary_mm K={k} N={n} {'x int' if args.xint else 'x randn'}")
        for fmt in args.fmts.split(","):
            w = ttnn.to_device(ttnn.from_torch(shuffled, dtype=dtypes[fmt], layout=ttnn.TILE_LAYOUT),
                               device, memory_config=a3.weight_mem(device, k, n))
            mb = weight_mb(n, k, fmt)
            for wpb in [int(v) for v in args.wpb.split(",")]:
                workers = a3.worker_cores(device, wpb)
                cores = [c for c, _, _ in workers]
                pcn = n // 32 // len(workers)
                xt = ttnn.from_torch(x.repeat(len(workers), 1).reshape(1, 1, len(workers), k), dtype=ttnn.bfloat16,
                                     layout=ttnn.TILE_LAYOUT, tile=tile, device=device,
                                     memory_config=a3.act_mem(cores, k))
                for sbk in [int(s) for s in args.subblock_k.split(",")]:
                    if (k // 32) % sbk:
                        continue
                    yt = ttnn.from_torch(torch.full((1, 1, 1, n), -7.0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                         tile=tile, device=device, memory_config=a3.out_mem(cores, n))
                    try:
                        a3.ternary_mm(xt, w, yt, workers, sbk)
                        ttnn.synchronize_device(device)
                        raw = ttnn.to_torch(yt).double().flatten().numpy()
                        y = assemble(raw, workers, pcn)
                        np.savez(os.path.expanduser(f"~/avan/native2bit/agentA/tmp/a3_{fmt}_{wpb}.npz"), raw=raw, ref=ref,
                                 workers=np.array([(c.x, c.y, b, s) for c, b, s in workers]))
                        pcc = float(np.corrcoef(y, ref)[0, 1])
                        rel = float(np.abs(y - ref).max() / np.abs(ref).max())
                        device_kernel_us(ttnn, device)  # drain
                        for _ in range(args.calls):
                            a3.ternary_mm(xt, w, yt, workers, sbk)
                        ttnn.synchronize_device(device)
                        devs = all_kernel_us(ttnn, device)
                        dev = float(np.median(devs))
                        print(f"{fmt:5s} wpb={wpb} sbk={sbk:3d} {mb:7.2f} MB {dev:7.1f} us {mb * 1e3 / dev:6.1f} GB/s "
                              f"pcc {pcc:.7f} maxrel {rel:.2e} n={len(devs)} [{min(devs):.1f}-{max(devs):.1f}]",
                              flush=True)
                    except Exception as e:  # noqa: BLE001
                        print(f"{fmt:5s} wpb={wpb} sbk={sbk:3d} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
                        rc = 1
                    ttnn.deallocate(yt)
                ttnn.deallocate(xt)
            ttnn.deallocate(w)
    finally:
        ttnn.close_device(device)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
