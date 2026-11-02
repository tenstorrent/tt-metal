# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Why is the LM head 8.1 ms in the op benchmark and ~3.5 ms in the model profile?

``a1_bound`` measures the stock matmul on ``lm_head`` (K=5120, N=248320) at
8.1 ms with a bfloat8_b weight -- 1.35 GB, i.e. 167 GB/s, 33% of peak. The
plan's decode profile lists the LM head at about 3.5 ms (385 GB/s). One of the
two is not measuring what it claims to, and the LM head is the largest single
tensor in the model, so a 4.6 ms/token error would distort the whole budget.

The difference has to be in the call, not the shape, so this sweeps the things a
call can differ in:

  * output memory config (DRAM, as the model's default, vs L1)
  * whether the input is in L1 (a model decode activation) or DRAM
  * N, to find a cliff: 16384, 62208, 124160, 248320 -- multiples of the 110-core
    grid behave differently, and 248320/110 is not an integer
  * the weight as bfloat8_b (model) vs bfloat4_b

If the time is flat in N up to some point and then jumps, the cause is the
program config's core/tile division rather than bandwidth, and the fix is a
program config, not a format.

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.a1_lmhead
"""

from __future__ import annotations

import argparse
import os

import torch

from .record import DRAM_PEAK_GBS
from .timing import Timer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5120)
    ap.add_argument("--ns", default="16384,62208,124160,248320")
    ap.add_argument("--formats", default="bfp8,bfp4")
    ap.add_argument("--calls", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    import ttnn

    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    timer = Timer(warmup=2, calls=args.calls, repeat=args.repeat)
    dt = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
    k = args.k

    print(f"Peak {DRAM_PEAK_GBS} GB/s, M=1, K={k}")
    print(f"{'N':>8} {'fmt':5s} {'MB':>8} {'out':5s} {'xin':4s} {'us':>9} {'GB/s':>7} {'%peak':>6}")
    try:
        for n in [int(x) for x in args.ns.split(",")]:
            w_src = torch.randn((k, n), dtype=torch.bfloat16)
            for fmt in args.formats.split(","):
                wt = ttnn.from_torch(
                    w_src, dtype=dt[fmt], layout=ttnn.TILE_LAYOUT,
                    device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                nb = n * k + n * k // 16 if fmt == "bfp8" else n * k // 2 + n * k // 16
                for out_name, out_mem in (("dram", ttnn.DRAM_MEMORY_CONFIG),
                                          ("l1", ttnn.L1_MEMORY_CONFIG)):
                    for in_name, in_mem in (("l1", ttnn.L1_MEMORY_CONFIG),
                                            ("dram", ttnn.DRAM_MEMORY_CONFIG)):
                        try:
                            x = ttnn.from_torch(
                                torch.randn((1, 1, 1, k), dtype=torch.bfloat16),
                                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                device=device, memory_config=in_mem,
                            )
                        except Exception as e:  # noqa: BLE001
                            print(f"{n:8d} {fmt:5s} alloc x/{in_name} FAILED {str(e)[:60]}")
                            continue

                        def once():
                            y = ttnn.linear(x, wt, memory_config=out_mem)
                            ttnn.synchronize_device(device)
                            ttnn.deallocate(y)

                        try:
                            med, _ = timer.best(once)
                            gbs = nb / med / 1e9
                            print(f"{n:8d} {fmt:5s} {nb / 1e6:8.2f} {out_name:5s} {in_name:4s} "
                                  f"{med * 1e6:9.1f} {gbs:7.1f} {gbs / DRAM_PEAK_GBS * 100:5.1f}%",
                                  flush=True)
                        except Exception as e:  # noqa: BLE001
                            print(f"{n:8d} {fmt:5s} {out_name:5s} {in_name:4s} FAILED "
                                  f"{type(e).__name__}: {str(e)[:70]}", flush=True)
                        finally:
                            ttnn.deallocate(x)
                ttnn.deallocate(wt)
            del w_src
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
