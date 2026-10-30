# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Why is A0 at 1.4 GB/s?  Time the reference test's own configuration.

``test_dram_matmul_compressed_bfp8`` runs [1, 7168] x [7168, 2048] with all-BFP8
tiles and passes, but nobody recorded how long it takes.  This runs exactly that
shape and tile format through the same code path and times it, then the same
shape as uniform BFP2, so the comparison is apples to apples:

  * if BFP8 [7168, 2048] is fast, the op is fine and my A0 harness is wrong
  * if it is also ~1.4 GB/s, the op is latency-bound at these shapes and the
    plan's 61 tok/s roofline does not rest on this code path

Also prints the achieved bandwidth of a plain ``ttnn.linear`` on the same
weights, which is the number the plan quotes (about 34% of roofline), so there
is a second, independent calibration point.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import numpy as np
import torch

DRAM_PEAK_GBS = 512.0


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def time_calls(fn, batch: int, pairs: int) -> float:
    """Marginal seconds per call: (t(2N) - t(N)) / N, median over pairs."""
    fn()  # warm
    samples = []
    for _ in range(pairs):
        t0 = time.perf_counter()
        for _ in range(batch):
            fn()
        t1 = time.perf_counter()
        for _ in range(batch):
            fn()
        t2 = time.perf_counter()
        samples.append(((t2 - t1) - (t1 - t0)) / batch)
    return float(statistics.median(samples))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=7168)
    ap.add_argument("--N", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--cores-per-bank", type=int, default=1)
    ap.add_argument("--subblock-k", type=int, default=0)
    args = ap.parse_args()

    import ttnn
    from models.demos.deepseek_v3_b1.compressed_tensor import CompressedTensor
    from models.demos.deepseek_v3_b1.micro_ops.dram_streaming_matmul_compressed.op import (
        DRAMStreamingMatmulCompressed,
    )
    from models.demos.deepseek_v3_b1.tests.unit_tests.test_dram_streaming_matmul import shuffle_tensor_tiles

    TILE = 32
    K, N = args.K, args.N
    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    try:
        primary = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
        nbanks = len(primary)
        cpb = args.cores_per_bank
        step = TILE * nbanks * cpb
        n_pad = ((N + step - 1) // step) * step
        per_core_n = n_pad // (nbanks * cpb)
        kt = K // TILE
        sbk = args.subblock_k or (kt // 4)
        sbk = max(2, sbk - (sbk % 2))

        print(f"K={K} N={N} banks={nbanks} cpb={cpb} n_pad={n_pad} per_core_n={per_core_n} kt={kt} sbk={sbk}")

        compute = []
        for p in primary:
            for off in range(cpb):
                compute.append(ttnn.CoreCoord(p.x + off, p.y))
        compute_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in compute])

        dram_grid = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
                )
            ]
        )
        b_shard = ttnn.ShardSpec(dram_grid, [K, per_core_n * cpb], ttnn.ShardOrientation.ROW_MAJOR)
        b_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, b_shard)

        m = 1
        a_torch = torch.randn((m, K), dtype=torch.bfloat16).repeat(len(compute), 1)
        a_shard = ttnn.ShardSpec(compute_set, [m, K], ttnn.ShardOrientation.ROW_MAJOR)
        a_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, a_shard)
        ttnn_a = ttnn.from_torch(
            a_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=a_mem,
            tile=ttnn.Tile([m, TILE]),
        )
        out_shard = ttnn.ShardSpec(compute_set, [m, per_core_n], ttnn.ShardOrientation.ROW_MAJOR)
        out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, out_shard)

        torch.manual_seed(0)
        w = torch.randn((K, N)).float()
        w_shuf = shuffle_tensor_tiles(w, TILE, nbanks)

        # --- timing helper for one tile format ---
        def run_case(label: str, fmt_index: int, tile_bytes: int):
            if n_pad != N:
                wpad = torch.nn.functional.pad(w_shuf, (0, n_pad - N))
            else:
                wpad = w_shuf
            assignment = np.full((kt, n_pad // TILE), fmt_index, dtype=np.int8)
            ct = CompressedTensor.from_bspm(wpad, assignment, device=device, memory_config=b_mem)
            out = ttnn.from_torch(
                torch.zeros((m, n_pad), dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=out_mem,
                tile=ttnn.Tile([m, TILE]),
            )

            def once():
                DRAMStreamingMatmulCompressed.op(ttnn_a, ct, out, subblock_k=sbk, cores_per_bank=cpb)
                ttnn.synchronize_device(device)

            us = time_calls(once, args.batch, args.pairs) * 1e6
            nbytes = kt * (n_pad // TILE) * tile_bytes
            gbs = nbytes / (us * 1e-6) / 1e9
            print(
                f"  {label:12s} {us:9.1f} us  {nbytes / 1e6:7.2f} MB  {gbs:7.2f} GB/s "
                f"({gbs / DRAM_PEAK_GBS * 100:5.1f}% peak)"
            )
            return ct, out

        print("DRAMStreamingMatmulCompressed, same configuration as the passing unit test:")
        run_case("bfp8", 0, 1088)
        run_case("bfp4", 1, 576)
        run_case("bfp2", 2, 320)

        # --- calibration: plain ttnn.linear on the same weights ---
        wt = ttnn.from_torch(
            w.bfloat16().T.contiguous(),  # [in, out]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x = ttnn.from_torch(
            torch.randn((1, 1, 1, K), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        def lin():
            y = ttnn.linear(x, wt, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.synchronize_device(device)
            ttnn.deallocate(y)

        us = time_calls(lin, args.batch, args.pairs) * 1e6
        bf16_bytes = K * N * 2
        print(
            f"  {'ttnn.linear':12s} {us:9.1f} us  {bf16_bytes / 1e6:7.2f} MB  "
            f"{bf16_bytes / (us * 1e-6) / 1e9:7.2f} GB/s (bf16 weights)"
        )
        print(f"\npeak assumed {DRAM_PEAK_GBS} GB/s")
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
