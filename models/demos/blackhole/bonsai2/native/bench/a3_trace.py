# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-op cost under trace: replay N calls of TernaryLinear (prep + A3) vs a stock ttnn.linear of the
same shape, wall time per call. Device kernel times are known from the profiles (prep 9.6 + A3 26 us
for ssm_out); the difference is what dispatch adds per op inside a trace."""
import os
import time

import torch


def main():
    import ttnn
    from models.demos.blackhole.bonsai2.native.a3.linear import TernaryLinear
    from models.demos.blackhole.bonsai2.native.a3.mlp import Shared

    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576,
                              trace_region_size=64 << 20, num_command_queues=1)
    try:
        sh = Shared(device)
        lin = TernaryLinear(sh, "blk.0.ssm_out.weight", os.path.expanduser("~/avan/native2bit/agentA/tt_cache/pq2_0/a3_linear"))
        x = ttnn.from_torch(torch.randn(1, 1, 1, 6144), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                            memory_config=ttnn.L1_MEMORY_CONFIG)
        w = ttnn.from_torch(torch.randn(6144, 5120), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        from models.demos.blackhole.bonsai2.native.a3.linear import gdn_in_rows, gdn_in_shape

        lin2 = TernaryLinear(sh, "blk.0.gdn_in", os.path.expanduser("~/avan/native2bit/agentA/tt_cache/pq2_0/a3_linear"),
                             rows=lambda: gdn_in_rows(sh, 0), shape=gdn_in_shape(sh, 0))
        x2 = ttnn.from_torch(torch.randn(1, 1, 1, 5120), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                             memory_config=ttnn.L1_MEMORY_CONFIG)

        def alternate():
            a = lin.forward(x)
            b = lin2.forward(x2)
            c = ttnn.add(x, x)
            ttnn.deallocate(a)
            ttnn.deallocate(c)
            return b

        n = 50
        for name, f in (("ternary prep+A3", lambda: lin.forward(x)), ("stock linear bfp8", lambda: ttnn.linear(x, w)),
                        ("ssm_out|gdn_in|add alternating", alternate), ("gdn_in alone", lambda: lin2.forward(x2)),
                        ("add alone", lambda: ttnn.add(x, x))):
            ttnn.deallocate(f())
            ttnn.synchronize_device(device)
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            outs = [f() for _ in range(n)]
            ttnn.end_trace_capture(device, tid, cq_id=0)
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            t0 = time.perf_counter()
            for _ in range(5):
                ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            dt = (time.perf_counter() - t0) / (5 * n) * 1e6
            print(f"{name:20s} traced wall per call {dt:7.1f} us", flush=True)
            ttnn.release_trace(device, tid)
            for o in outs:
                ttnn.deallocate(o)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
