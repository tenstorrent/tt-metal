# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stock-format baseline for A3: deepseek_v3_b1's DRAMStreamingMatmul on Bonsai shapes.

That op is the natural skeleton for the fused-scale kernel: in0 is a 1x32 tiny tile
replicated on each worker (no mcast), in1 is DRAM-sharded and streamed as K-sticks
per output column, one worker per DRAM bank, all in a generic_op with JIT kernels.
Before forking it, measure it as is with ternary codes in bfp2/bfp4/bfp8 against
the stock ttnn DRAM-sharded numbers (a1_bfp2_shard.py: down bfp2 83.2 / bfp4 121.2).

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.a3_dsmm [--k 5120 --n 17408]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .a1_bfp2_shard import all_kernel_us, weight_mb
from .a1_calib import device_kernel_us


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=17408)
    ap.add_argument("--n", type=int, default=5120)
    ap.add_argument("--fmts", default="bfp2,bfp4,bfp8")
    ap.add_argument("--subblock-k", default="4,8,16,34")
    ap.add_argument("--calls", type=int, default=9)
    ap.add_argument("--fidelity", default="LoFi")
    ap.add_argument("--xint", action="store_true", help="x = small integers (exact at any fidelity)")
    args = ap.parse_args()
    for key, val in (("TT_METAL_DEVICE_PROFILER", "1"), ("TT_METAL_PROFILER_CPP_POST_PROCESS", "1"),
                     ("TT_METAL_PROFILER_MID_RUN_DUMP", "1")):
        os.environ.setdefault(key, val)

    import ttnn
    from models.demos.deepseek_v3_b1.micro_ops.dram_streaming_matmul.op import DRAMStreamingMatmul
    from models.demos.deepseek_v3_b1.tests.unit_tests.test_dram_streaming_matmul import shuffle_tensor_tiles

    k, n = args.k, args.n
    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    dtypes = {"bfp2": ttnn.DataType.BFLOAT2_B, "bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b}
    rc = 0
    try:
        cores = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
        banks = device.dram_grid_size().x
        assert len(cores) == banks and n % (32 * banks) == 0
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in cores])
        tile = ttnn.Tile([1, 32])
        x = torch.randint(-7, 8, (1, 1, 1, k)).float() if args.xint else torch.randn(1, 1, 1, k).bfloat16().float()
        codes = torch.from_numpy(np.random.default_rng(0).integers(-1, 2, (k, n)).astype(np.float32))
        ref = (x[0, 0].double() @ codes.double()).flatten().numpy()
        x_t = ttnn.from_torch(
            x.repeat(1, 1, banks, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, tile=tile,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                                            ttnn.ShardSpec(grid, [1, k], ttnn.ShardOrientation.ROW_MAJOR)))
        out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
                                    ttnn.ShardSpec(grid, (1, n // banks), ttnn.ShardOrientation.ROW_MAJOR))
        w_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM,
                                  ttnn.ShardSpec(ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0),
                                                                                   ttnn.CoreCoord(banks - 1, 0))}),
                                                 [k, n // banks], ttnn.ShardOrientation.ROW_MAJOR))
        shuffled = shuffle_tensor_tiles(codes.reshape(1, 1, k, n), 32, banks)
        print(f"DRAMStreamingMatmul K={k} N={n}, {banks} workers, in0 1x32, {args.fidelity} fp32acc")
        for fmt in args.fmts.split(","):
            w_t = ttnn.to_device(ttnn.from_torch(shuffled, dtype=dtypes[fmt], layout=ttnn.TILE_LAYOUT),
                                 device, memory_config=w_mem)
            mb = weight_mb(n, k, fmt)
            for sbk in [int(s) for s in args.subblock_k.split(",")]:
                if (k // 32) % sbk:
                    continue
                y_t = ttnn.from_torch(torch.zeros(1, 1, 1, n), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                      device=device, tile=tile, memory_config=out_mem)
                try:
                    def once():
                        DRAMStreamingMatmul.op(x_t, w_t, y_t, fp32_dest_acc_en=True,
                                               math_fidelity=getattr(ttnn.MathFidelity, args.fidelity), subblock_k=sbk)
                    once()
                    ttnn.synchronize_device(device)
                    y = ttnn.to_torch(y_t).double().flatten().numpy()[:n]
                    pcc = float(np.corrcoef(y, ref)[0, 1])
                    terr = np.abs(y - ref).reshape(-1, 32).max(1)
                    bad = np.nonzero(terr > 1e-2 * np.abs(ref).max())[0]
                    print(f"  bad tiles {len(bad)}/{len(terr)}: {bad[:40].tolist()}", flush=True)
                    if len(bad):
                        t = bad[0]
                        print(f"  tile {t} got {y[t*32:t*32+6].round(1).tolist()} ref {ref[t*32:t*32+6].round(1).tolist()}", flush=True)
                    device_kernel_us(ttnn, device)  # drain
                    for _ in range(args.calls):
                        once()
                    ttnn.synchronize_device(device)
                    devs = all_kernel_us(ttnn, device)
                    dev = float(np.median(devs))
                    print(f"{fmt:5s} sbk={sbk:3d} {mb:7.2f} MB {dev:7.1f} us {mb * 1e3 / dev:6.1f} GB/s "
                          f"pcc {pcc:.7f} n={len(devs)} [{min(devs):.1f}-{max(devs):.1f}]", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"{fmt:5s} sbk={sbk:3d} FAILED {type(e).__name__}: {str(e)[:150]}", flush=True)
                    rc = 1
                ttnn.deallocate(y_t)
            ttnn.deallocate(w_t)
    finally:
        ttnn.close_device(device)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
