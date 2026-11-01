# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stage times of prefill_mlp (wall ms, synchronized, median of 5) on layer 0, and the same matmuls
with bfp4 weights (the dense path's gate/up dtype).

Usage: python -m models.demos.blackhole.bonsai2.native.a3.bench_prefill [--seqs 128,2048]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="128,2048")
    args = ap.parse_args()
    import ttnn
    from models.demos.blackhole.bonsai2.native.a3 import prefill as pf
    from models.demos.blackhole.bonsai2.native.a3.mlp import Shared, TernaryMLP

    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)

    def timed(f, n=5):
        ttnn.deallocate(f())
        ttnn.synchronize_device(device)
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            r = f()
            ttnn.synchronize_device(device)
            ts.append((time.perf_counter() - t0) * 1e3)
            ttnn.deallocate(r)
        return float(np.median(ts))

    try:
        sh = Shared(device)
        rot = pf.rotator(sh)
        mlp = TernaryMLP(sh, 0, os.path.join(os.environ["TT_CACHE_PATH"], "a3_mlp"))
        wg, wu = pf.dequant(mlp.w_gu, mlp.s_gu, 5120, 17408, gu=True)
        wd = pf.dequant(mlp.w_d, mlp.s_d, 17408, 5120)
        wg4 = ttnn.typecast(wg, ttnn.bfloat4_b)
        wd4 = ttnn.typecast(wd, ttnn.bfloat4_b)
        lofi = pf._LOFI
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        pc = lambda q, k, n: tpc.create_prefill_matmul_program_config(  # noqa: E731
            q, k, n, grid_size=tpc.prefill_grid_default(), tuning=tpc.prefill_tuning(1))
        for s in [int(v) for v in args.seqs.split(",")]:
            for mem_name, mem in (("dram", ttnn.DRAM_MEMORY_CONFIG), ("l1", ttnn.L1_MEMORY_CONFIG)):
                if mem_name == "l1" and s > 512:
                    continue
                x = ttnn.from_torch(torch.randn(1, 1, s, 5120), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                    device=device, memory_config=mem)
                h = ttnn.from_torch(torch.randn(1, 1, s, 17408), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                    device=device, memory_config=mem)
                x3 = ttnn.reshape(x, [1, s, 5120])
                h3 = ttnn.reshape(h, [1, s, 17408])
                r = {
                    "rot5120": timed(lambda: rot(x, mem)),
                    "rot17408": timed(lambda: rot(h, mem)),
                    "gate_bfp8": timed(lambda: ttnn.linear(x, wg, compute_kernel_config=lofi, memory_config=mem)),
                    "gate_bfp4": timed(lambda: ttnn.linear(x, wg4, compute_kernel_config=lofi, memory_config=mem)),
                    "down_bfp8": timed(lambda: ttnn.linear(h, wd, compute_kernel_config=lofi, memory_config=mem)),
                    "down_bfp4": timed(lambda: ttnn.linear(h, wd4, compute_kernel_config=lofi, memory_config=mem)),
                    "gate_bfp4_3d": timed(lambda: ttnn.linear(x3, wg4, compute_kernel_config=lofi, memory_config=mem)),
                    "gate_bfp8_3d": timed(lambda: ttnn.linear(x3, wg, compute_kernel_config=lofi, memory_config=mem)),
                    "down_bfp8_3d_pc": timed(lambda: ttnn.linear(h3, wd, compute_kernel_config=lofi, memory_config=mem,
                                                                 program_config=pc(s, 17408, 5120))),
                    "gate_bfp8_pc": timed(lambda: ttnn.linear(x, wg, compute_kernel_config=lofi, memory_config=mem,
                                                              program_config=pc(s, 5120, 17408))),
                    "gate_bfp4_pc": timed(lambda: ttnn.linear(x, wg4, compute_kernel_config=lofi, memory_config=mem,
                                                              program_config=pc(s, 5120, 17408))),
                    "mul": timed(lambda: ttnn.mul(h, h, memory_config=mem)),
                    "mlp": timed(lambda: pf.prefill_mlp(mlp, x, mem)),
                }
                print(f"S={s} act={mem_name} wall ms: " + "  ".join(f"{k} {v:.3f}" for k, v in r.items()), flush=True)
                ttnn.deallocate(x)
                ttnn.deallocate(h)
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
