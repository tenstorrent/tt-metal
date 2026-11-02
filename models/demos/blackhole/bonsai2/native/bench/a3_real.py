# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A3 real-weight gate: GGUF PQ2_0 projections through ternary_mm_scaled vs cpu_ref.

Device input is bf16(H(D x)); two references:
  ref     = cpu_ref.project_ternary(t, had, x), float64 throughout (the plan's gate, PCC >= 0.9999)
  ref_bf  = the same with H(D x) rounded to bf16 first, which isolates the kernel's own error.

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.a3_real [--proj blk.0.ffn_down.weight,...]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .a1_bfp2_shard import all_kernel_us
from .a1_calib import device_kernel_us
from .a3_tmm import assemble

GGUF = os.path.expanduser("~/avan/models/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=GGUF)
    ap.add_argument("--proj", default="blk.0.ffn_down.weight,blk.0.ffn_gate.weight,blk.0.ffn_up.weight")
    ap.add_argument("--wpb", type=int, default=4)
    ap.add_argument("--calls", type=int, default=9)
    args = ap.parse_args()
    for key, val in (("TT_METAL_DEVICE_PROFILER", "1"), ("TT_METAL_PROFILER_CPP_POST_PROCESS", "1"),
                     ("TT_METAL_PROFILER_MID_RUN_DUMP", "1")):
        os.environ.setdefault(key, val)

    import ttnn
    from models.demos.blackhole.bonsai2.native import cpu_ref
    from models.demos.blackhole.bonsai2.native.a3 import op as a3
    from models.demos.blackhole.bonsai2.native.gguf_ternary import read_gguf, read_ternary
    from models.demos.blackhole.bonsai2.native.hadamard import BlockHadamard

    gguf = read_gguf(args.gguf)
    had = BlockHadamard.from_metadata(gguf.kv)
    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    rc = 0
    try:
        banks = device.dram_grid_size().x
        workers = a3.worker_cores(device, args.wpb)
        cores = [c for c, _, _ in workers]
        rng = np.random.default_rng(0)
        for name in args.proj.split(","):
            t = read_ternary(gguf, name)
            n, k = t.codes.shape
            x = rng.standard_normal((1, k)).astype(np.float32)
            ref = cpu_ref.project_ternary(t, had, x).flatten()
            xr = torch.from_numpy(had.rotate(x).astype(np.float32)).bfloat16().float()
            ref_bf = (xr.double().numpy() @ t.dequant().astype(np.float64).T).flatten()

            codes = torch.from_numpy(t.codes.T.astype(np.float32).copy())  # [K, N]
            w = ttnn.to_device(ttnn.from_torch(a3.shuffle_for_banks(codes.reshape(1, 1, k, n), banks),
                                               dtype=ttnn.DataType.BFLOAT2_B, layout=ttnn.TILE_LAYOUT),
                               device, memory_config=a3.weight_mem(device, k, n))
            srows = a3.scale_rows(torch.from_numpy(t.scales.T.astype(np.float32).copy()))  # [K/128, N] -> rows
            s = ttnn.to_device(ttnn.from_torch(a3.shuffle_for_banks(srows.reshape(1, 1, *srows.shape), banks),
                                               dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT),
                               device, memory_config=a3.weight_mem(device, srows.shape[0], n))
            x8 = a3.mask_x(xr.flatten())
            x8t = ttnn.from_torch(x8.repeat(len(workers), 1).reshape(1, 1, 8 * len(workers), k), dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile([8, 32]), device=device,
                                  memory_config=ttnn.MemoryConfig(
                                      ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                                      ttnn.ShardSpec(a3.core_set(cores), [8, k], ttnn.ShardOrientation.ROW_MAJOR)))
            yt = ttnn.from_torch(torch.full((1, 1, 1, n), -7.0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                 tile=ttnn.Tile([1, 32]), device=device, memory_config=a3.out_mem(cores, n))
            a3.ternary_mm_scaled(x8t, w, s, yt, workers)
            ttnn.synchronize_device(device)
            y = assemble(ttnn.to_torch(yt).double().flatten().numpy(), workers, n // 32 // len(workers))
            device_kernel_us(ttnn, device)
            for _ in range(args.calls):
                a3.ternary_mm_scaled(x8t, w, s, yt, workers)
            ttnn.synchronize_device(device)
            dev = float(np.median(all_kernel_us(ttnn, device)))
            p, pb = cpu_ref.pcc(y, ref), cpu_ref.pcc(y, ref_bf)
            ok = p >= 0.9999
            rc |= 0 if ok else 1
            print(f"{name:26s} [{n}, {k}] {dev:6.1f} us  pcc vs cpu_ref {p:.7f}  vs bf16-x ref {pb:.7f}  "
                  f"{'ok' if ok else 'FAIL'}", flush=True)
            for tt in (w, s, x8t, yt):
                ttnn.deallocate(tt)
            a3.clear_cache()
    finally:
        ttnn.close_device(device)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
