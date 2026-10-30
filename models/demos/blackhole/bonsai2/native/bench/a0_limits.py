# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What limits the compressed op: bytes, columns, or M?

A0 measured the DeepSeek compressed op at 1.3 GB/s on bfp2 and 4.4 GB/s on
bfp8 -- the same wall-clock time for 3.4x the bytes -- while ``ttnn.linear``
reaches 361 GB/s on the identical shape. So the op is not bandwidth-bound, and
the plan's BFP2 streaming roofline cannot be read off it as-is.

This isolates the cause with three orthogonal sweeps, all on one frozen
[K=5120, N] weight so only one variable moves at a time:

  columns   N varies, K and M fixed     -> cost per output column
  bytes     format varies, shape fixed  -> is it reading bytes at all?
  rows      M varies (1, 8, 32), shape fixed -> tiny-tile vs standard-tile

If time tracks *columns* and not bytes, the per-column synchronisation dominates
and the number of columns per core has to come down (more cores per bank, or a
kernel that batches columns). If time drops sharply with M, the M=1 tiny-tile
path is the problem and prefill-sized M is the efficient operating point.

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.a0_limits
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .a0_roofline import TILE, TILE_BYTES, FMT_INDEX, build_linear
from .record import DRAM_PEAK_GBS
from .timing import Timer


def build_M(ttnn, device, n: int, k: int, m: int, fmt: str, cpb: int, subblock_k: int, seed: int = 0):
    """The compressed op with an explicit M, so the tile height can be swept."""
    from models.demos.deepseek_v3_b1.compressed_tensor import CompressedTensor
    from models.demos.deepseek_v3_b1.micro_ops.dram_streaming_matmul_compressed.op import (
        DRAMStreamingMatmulCompressed,
    )
    from models.demos.deepseek_v3_b1.tests.unit_tests.test_dram_streaming_matmul import shuffle_tensor_tiles

    primary = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    nbanks = len(primary)
    step = TILE * nbanks * cpb
    n_pad = ((n + step - 1) // step) * step
    per_core_tiles = n_pad // (nbanks * cpb) // TILE
    kt = k // TILE

    g = torch.Generator().manual_seed(seed)
    w = torch.randint(-1, 2, (k, n), generator=g).float()
    if n_pad != n:
        w = torch.nn.functional.pad(w, (0, n_pad - n))
    w = shuffle_tensor_tiles(w, TILE, nbanks)

    compute = [ttnn.CoreCoord(p.x + off, p.y) for p in primary for off in range(cpb)]
    compute_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in compute])
    dram_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
            )
        ]
    )
    b_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_grid, [k, per_core_tiles * TILE * cpb], ttnn.ShardOrientation.ROW_MAJOR),
    )
    ct = CompressedTensor.from_bspm(
        w, np.full((kt, n_pad // TILE), FMT_INDEX[fmt], dtype=np.int8), device=device, memory_config=b_mem
    )

    a_torch = torch.randn((m, k), dtype=torch.bfloat16).repeat(len(compute), 1)
    ttnn_a = ttnn.from_torch(
        a_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(compute_set, [m, k], ttnn.ShardOrientation.ROW_MAJOR),
        ),
        tile=ttnn.Tile([m, TILE]),
    )
    out = ttnn.from_torch(
        torch.zeros((m, n_pad), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(compute_set, [m, per_core_tiles * TILE], ttnn.ShardOrientation.ROW_MAJOR),
        ),
        tile=ttnn.Tile([m, TILE]),
    )
    keep = ct

    def once():
        DRAMStreamingMatmulCompressed.op(ttnn_a, keep, out, subblock_k=subblock_k, cores_per_bank=cpb)
        ttnn.synchronize_device(device)

    nbytes = kt * (n_pad // TILE) * TILE_BYTES[fmt]
    return once, nbytes, n_pad, per_core_tiles


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    import ttnn

    K = 5120
    timer = Timer(warmup=2, calls=args.calls, repeat=args.repeat)
    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    try:
        print(f"peak assumed {DRAM_PEAK_GBS} GB/s; K={K} fixed for all rows\n")

        print("columns sweep (M=1, bfp2, cpb=1, sbk=16):")
        base = None
        for n in (256, 1024, 4096, 16384):
            fn, nbytes, n_pad, pcnt = build_M(ttnn, device, n, K, 1, "bfp2", 1, 16)
            med, _ = timer.best(fn)
            us_per_col = med * 1e6 / n_pad
            if base is None:
                base = us_per_col
            print(
                f"  N={n:6d}  {med * 1e6:9.1f} us  {nbytes / 1e6:7.2f} MB  "
                f"{nbytes / med / 1e9:7.1f} GB/s   {us_per_col:7.4f} us/col  ({pcnt} tiles/core)"
            )

        print("\nbytes sweep (M=1, N=4096, cpb=1, sbk=16):")
        for fmt in ("bfp2", "bfp4", "bfp8"):
            fn, nbytes, n_pad, pcnt = build_M(ttnn, device, 4096, K, 1, fmt, 1, 16)
            med, _ = timer.best(fn)
            print(f"  {fmt:5s} {med * 1e6:9.1f} us  {nbytes / 1e6:7.2f} MB  {nbytes / med / 1e9:7.1f} GB/s")

        print("\nrows sweep (N=4096, bfp2, cpb=1, sbk=16):")
        for m in (1, 2, 8, 32):
            fn, nbytes, n_pad, pcnt = build_M(ttnn, device, 4096, K, m, "bfp2", 1, 16)
            med, _ = timer.best(fn)
            print(
                f"  M={m:2d}  {med * 1e6:9.1f} us  {nbytes / med / 1e9:7.1f} GB/s  "
                f"({nbytes / med / 1e9 / DRAM_PEAK_GBS * 100:.1f}% peak)"
            )

        print("\nttnn.linear control (N=4096, K=5120, bf16):")
        code = np.random.default_rng(0).integers(-1, 2, size=(4096, K)).astype(np.int8)
        fn, nbytes = build_linear(ttnn, device, code)
        med, _ = timer.best(fn)
        print(f"  linear {med * 1e6:9.1f} us  {nbytes / 1e6:7.2f} MB  {nbytes / med / 1e9:7.1f} GB/s")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
