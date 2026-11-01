# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Control: does the same single-tile matmul work for bfloat4_b, bfloat8_b, and bf16?

The bf2 single-tile result (PCC 0.0048, device |y| max 13.0 against a reference
10.49) looks like a permutation, but the layout sweep that tried to confirm that
was worthless -- its permutation loop was a no-op, so all eight candidates produced
identical bytes and identical PCC. A test that cannot change its input cannot
distinguish anything.

Before interpreting the bf2 number at all, this establishes the baseline: the same
32x32 matmul, same activation, same call, for bf16 / bf8 / bf4 / bf2. Whatever bf4
does is the behaviour a 4-bit block float gets here; if bf4 matches its reference
and bf2 does not, the fault is specific to Bfp2_b. If bf4 also fails at 32x32, then
32x32 is simply too small for block-float tiles on this path and the bf2 result at
that size says nothing.

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.a1_bfp2_control
"""

from __future__ import annotations

import os

import numpy as np
import torch


def pcc(a, b) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main() -> int:
    import ttnn

    device = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    rng = np.random.default_rng(11)
    # Ternary weights so every dtype can represent them exactly: any difference in
    # the result is then about the format's plumbing, not about rounding.
    codes = rng.integers(-1, 2, (32, 32)).astype(np.float32)
    x1 = rng.standard_normal((1, 32)).astype(np.float32)
    ref = x1.astype(np.float64) @ codes.astype(np.float64).T

    dtypes = [("bf16", ttnn.bfloat16), ("bfp8", ttnn.bfloat8_b),
              ("bfp4", ttnn.bfloat4_b), ("bfp2", ttnn.DataType.BFLOAT2_B)]
    rc = 0
    try:
        print(f"single 32x32 tile, ternary weights, reference |y| max {np.abs(ref).max():.4f}")
        print(f"{'dtype':6s} {'pcc':>10s} {'dev |y| max':>12s} {'verdict':>10s}")
        for name, dt in dtypes:
            try:
                wt = ttnn.from_torch(torch.from_numpy(codes), dtype=dt,
                                     layout=ttnn.TILE_LAYOUT, device=device,
                                     memory_config=ttnn.DRAM_MEMORY_CONFIG)
                xt = ttnn.from_torch(torch.from_numpy(x1).to(torch.bfloat16).reshape(1, 1, 1, 32),
                                     dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                                     memory_config=ttnn.L1_MEMORY_CONFIG)
                y = ttnn.linear(xt, wt, memory_config=ttnn.L1_MEMORY_CONFIG)
                got = ttnn.to_torch(y).float().numpy().reshape(32)
                ttnn.deallocate(y)
                ttnn.deallocate(xt)
                ttnn.deallocate(wt)
                c = pcc(got, ref[0])
                verdict = "ok" if c > 0.999 else ("WRONG" if c > 0.01 else "garbage")
                print(f"{name:6s} {c:10.7f} {np.abs(got).max():12.4f} {verdict:>10s}", flush=True)
                if c <= 0.999:
                    rc = 1
            except Exception as e:  # noqa: BLE001
                print(f"{name:6s} FAILED {type(e).__name__}: {str(e)[:110]}", flush=True)
                rc = 1
        print()
        print("Read: bf4 is the control. If bf4 matches and bf2 does not, the fault is in")
        print("Bfp2_b specifically. If bf4 also fails at 32x32, this shape is too small")
        print("for block-float tiles here and the bf2 result at this size means nothing.")
    finally:
        ttnn.close_device(device)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
