# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""If bfp2 decodes exactly, where does PCC 0.883 come from?

The one-hot probe showed bfp2 reproduces the weight bit-exactly through the matmul
path (0/1024 values wrong, same as bf4). Yet the ffn_down comparison reports PCC
0.883 against a float64 reference. Both cannot be about the device: the weight the
device sees is right, so the discrepancy is in how the reference was built.

The suspects, all in the bench rather than in tt-metal:

  1. **The scales.** The reference folds a per-128-group scale into the weight, but
     the matmul under test is given the *codes only* -- no scale is applied on
     device. So the reference and the device are computing different products. This
     is the most likely cause and would produce a plausible-looking correlation
     rather than nonsense, exactly as observed.
  2. **Activation dtype.** x is bf16 on device and float32 in the reference.
  3. **Accumulation.** fp32 dest accumulation on device versus float64 on host.

The test: run the same ffn_down comparison three ways -- codes-only against
codes-only (the honest control), codes-with-scales against scaled reference (the
intended A1-stock semantics), and bfp4 through the identical harness. If the
codes-only control matches exactly, the 0.883 was a broken comparison and the
format was never at fault.

Usage::

    python -m models.demos.blackhole.bonsai2.native.bench.a1_bfp2_recheck
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
    # A shape big enough to have many tiles and several scale groups, small enough
    # to keep host reference work instant.
    k, n, group = 5120, 512, 128
    rng = np.random.default_rng(0)
    codes = rng.integers(-1, 2, (n, k)).astype(np.float32)
    x_np = rng.standard_normal((1, k)).astype(np.float32)
    rc = 0
    try:
        x = ttnn.from_torch(
            torch.from_numpy(x_np).to(torch.bfloat16).reshape(1, 1, 1, k),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG)
        x_eff = torch.from_numpy(x_np).to(torch.bfloat16).float().numpy()

        for label, dt in (("bfp4", ttnn.bfloat4_b), ("bfp2", ttnn.DataType.BFLOAT2_B)):
            wt = ttnn.from_torch(torch.from_numpy(codes.T.copy()), dtype=dt,
                                 layout=ttnn.TILE_LAYOUT, device=device,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
            y = ttnn.linear(x, wt, memory_config=ttnn.L1_MEMORY_CONFIG)
            got = ttnn.to_torch(y).float().numpy().reshape(1, n)
            ttnn.deallocate(y)
            ttnn.deallocate(wt)

            # (1) codes only, no scales -- what the matmul was actually given.
            ref_codes = x_eff.astype(np.float64) @ codes.astype(np.float64).T
            c1 = pcc(got, ref_codes)
            # (2) the old, wrong comparison: reference with per-group scales folded
            # in, which the device was never asked to apply.
            scales = rng.uniform(1e-3, 2e-2, (n, k // group)).astype(np.float64)
            scaled = (codes.astype(np.float64) * np.repeat(scales, group, axis=1)).T
            ref_scaled = x_eff.astype(np.float64) @ scaled
            c2 = pcc(got, ref_scaled)
            # (3) exactness, not just correlation.
            maxabs = float(np.abs(got - ref_codes).max())
            relmax = maxabs / (np.abs(ref_codes).max() + 1e-30)
            exact = int((np.abs(got - ref_codes) <= 1e-3).sum())
            print(f"{label}: codes-only PCC {c1:.7f}  max|err| {maxabs:.5f} "
                  f"(rel {relmax:.2e}, {exact}/{got.size} within 1e-3)")
            print(f"       scaled-reference PCC {c2:.7f}  <- the number a scaled reference "
                  f"reports when the device applied no scale")
            if c1 < 0.9999:
                print(f"       -> codes-only comparison FAILS: the matmul really differs")
                rc = 1
        print()
        print("If codes-only matches and scaled does not, the 0.883 was a broken")
        print("comparison: the device was never applying the per-group scales, so a")
        print("scaled reference was measuring the scales' absence, not the format.")
        ttnn.deallocate(x)
    finally:
        ttnn.close_device(device)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
