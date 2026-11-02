# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Which stock ttnn ops take the A3 op's 1x32 tiny-tile outputs? (glue feasibility probe)"""
import os
import time

import torch


def main():
    import ttnn
    from models.demos.blackhole.bonsai2.native.a3 import op as a3

    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    try:
        n = 17408
        workers = a3.worker_cores(device, 4)
        cores = [c for c, _, _ in workers]
        g = torch.randn(1, 1, 1, n)
        u = torch.randn(1, 1, 1, n)
        mk = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile([1, 32]),  # noqa: E731
                                       device=device, memory_config=a3.out_mem(cores, n))
        gt, ut = mk(g), mk(u)

        def tryit(name, f):
            try:
                ttnn.synchronize_device(device)
                t0 = time.perf_counter()
                for _ in range(20):
                    r = f()
                ttnn.synchronize_device(device)
                dt = (time.perf_counter() - t0) / 20 * 1e6
                print(f"OK   {name:40s} {dt:7.1f} us  shape {tuple(r.shape)} tile {r.get_tile().tile_shape} "
                      f"layout {r.layout} mem {r.memory_config().memory_layout}", flush=True)
                return r
            except Exception as e:  # noqa: BLE001
                print(f"FAIL {name:40s} {type(e).__name__}: {str(e).splitlines()[0][:160]}", flush=True)
                return None

        il = tryit("to_memory_config sharded->L1 interleaved", lambda: ttnn.to_memory_config(gt, ttnn.L1_MEMORY_CONFIG))
        tryit("silu (sharded tiny)", lambda: ttnn.silu(gt))
        tryit("mul (sharded tiny)", lambda: ttnn.mul(gt, ut))
        tryit("mul silu fused (sharded tiny)", lambda: ttnn.mul(gt, ut, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]))
        if il is not None:
            tryit("silu (interleaved tiny)", lambda: ttnn.silu(il))
            rm = tryit("to_layout ROW_MAJOR (interleaved tiny)", lambda: ttnn.to_layout(il, ttnn.ROW_MAJOR_LAYOUT))
            if rm is not None:
                tryit("to_layout TILE 32x32 from RM", lambda: ttnn.to_layout(rm, ttnn.TILE_LAYOUT))
                tryit("tilize_with_zero_padding", lambda: ttnn.tilize_with_zero_padding(ttnn.reshape(rm, [1, 1, 1, n])))
        tryit("untilize (sharded tiny)", lambda: ttnn.untilize(gt))
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
