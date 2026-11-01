# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A3 exact-ternary MLP (native/a3/mlp.py) on real PQ2_0 layers vs a float64 cpu_ref chain.

ref: g = P(gate, x), u = P(up, x), h = silu(g) * u, y = P(down, h), P = cpu_ref.project_ternary.

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.a3_mlp [--layers 0,47]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .a1_bfp2_shard import all_kernel_us
from .a1_calib import device_kernel_us


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,47")
    ap.add_argument("--calls", type=int, default=5)
    args = ap.parse_args()
    for key, val in (("TT_METAL_DEVICE_PROFILER", "1"), ("TT_METAL_PROFILER_CPP_POST_PROCESS", "1"),
                     ("TT_METAL_PROFILER_MID_RUN_DUMP", "1")):
        os.environ.setdefault(key, val)

    import ttnn
    from models.demos.blackhole.bonsai2.native import cpu_ref
    from models.demos.blackhole.bonsai2.native.a3.mlp import Shared, TernaryMLP
    from models.demos.blackhole.bonsai2.native.gguf_ternary import read_ternary

    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    rc = 0
    try:
        sh = Shared(device)
        rng = np.random.default_rng(0)
        for layer in [int(v) for v in args.layers.split(",")]:
            mlp = TernaryMLP(sh, layer)
            x = rng.standard_normal((1, 5120)).astype(np.float32)
            xb = torch.from_numpy(x).bfloat16().float().numpy()
            t = {n: read_ternary(sh.gguf, f"blk.{layer}.ffn_{n}.weight") for n in ("gate", "up", "down")}
            g = cpu_ref.project_ternary(t["gate"], sh.had, xb)
            u = cpu_ref.project_ternary(t["up"], sh.had, xb)
            h = g / (1 + np.exp(-g)) * u
            ref = cpu_ref.project_ternary(t["down"], sh.had, h).flatten()
            xt = ttnn.from_torch(torch.from_numpy(np.repeat(xb, 32, 0)).reshape(1, 1, 32, 5120), dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
            y = ttnn.to_torch(mlp.forward(xt)).float().numpy()
            ttnn.synchronize_device(device)
            hd = ttnn.to_torch(sh.h).float().numpy().flatten()
            p_h, p_y = cpu_ref.pcc(hd, h.flatten()), cpu_ref.pcc(y[0, 0, 0], ref)
            rows_zero = bool(np.all(y[0, 0, 1:] == 0))
            device_kernel_us(ttnn, device)
            for _ in range(args.calls):
                ttnn.deallocate(mlp.forward(xt))
            ttnn.synchronize_device(device)
            d = np.array(all_kernel_us(ttnn, device)).reshape(args.calls, 4)
            st = np.median(d, axis=0)
            ok = p_y >= 0.9999
            rc |= 0 if ok else 1
            print(f"layer {layer:2d}: pcc h {p_h:.7f}  y {p_y:.7f}  rows 1-31 zero {rows_zero}  "
                  f"{'ok' if ok else 'FAIL'}\n   device us: prep5120 {st[0]:.1f}  gate|up {st[1]:.1f}  "
                  f"prep17408 {st[2]:.1f}  down {st[3]:.1f}  total {np.median(d.sum(1)):.1f}", flush=True)
    finally:
        ttnn.close_device(device)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
