# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Settle the orientation of ttnn.linear once, with bf16, so it stops costing runs.

Every failed comparison today could in principle be an orientation mistake, and one
of my "controls" reported bf16, bf8, bf4 and bf2 all at exactly the same PCC, which
means the comparison itself was wrong -- not any of the formats.

This is the smallest possible test: a random bf16 weight and activation, both known
exactly, compared against x @ w.T and x @ w. Whatever matches is the convention, and
it gets written down for the rest of the bench suite.
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
    w = rng.standard_normal((32, 32)).astype(np.float32)
    x = rng.standard_normal((1, 32)).astype(np.float32)
    try:
        wt = ttnn.from_torch(torch.from_numpy(w), dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, device=device)
        for label, shape in (("1x1x1x32", (1, 1, 1, 32)), ("1x32", (1, 32)), ("32", (32,))):
            xt = ttnn.from_torch(torch.from_numpy(x).to(torch.bfloat16).reshape(shape),
                                 dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                                 memory_config=ttnn.L1_MEMORY_CONFIG)
            try:
                y = ttnn.to_torch(ttnn.linear(xt, wt, memory_config=ttnn.L1_MEMORY_CONFIG))
                got = y.float().numpy().reshape(-1)
                print(f"x reshape {label:9s} out {tuple(y.shape)}: "
                      f"pcc vs x@w.T {pcc(got, (x @ w.T)[0]):.7f}   "
                      f"pcc vs x@w {pcc(got, (x @ w)[0]):.7f}")
            except Exception as e:  # noqa: BLE001
                print(f"x reshape {label:9s} FAILED {type(e).__name__}: {str(e)[:100]}")
            finally:
                ttnn.deallocate(xt)
        # And the same for a 512x512 weight, which is closer to the real shapes.
        w2 = rng.standard_normal((512, 512)).astype(np.float32)
        x2 = rng.standard_normal((1, 512)).astype(np.float32)
        wt2 = ttnn.from_torch(torch.from_numpy(w2), dtype=ttnn.bfloat16,
                              layout=ttnn.TILE_LAYOUT, device=device)
        xt2 = ttnn.from_torch(torch.from_numpy(x2).to(torch.bfloat16).reshape(1, 1, 1, 512),
                              dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                              memory_config=ttnn.L1_MEMORY_CONFIG)
        got2 = ttnn.to_torch(ttnn.linear(xt2, wt2, memory_config=ttnn.L1_MEMORY_CONFIG)).float().numpy().reshape(-1)
        print(f"512x512: pcc vs x@w.T {pcc(got2, (x2 @ w2.T)[0]):.7f}   "
              f"pcc vs x@w {pcc(got2, (x2 @ w2)[0]):.7f}")
        ttnn.deallocate(xt2)
        ttnn.deallocate(wt2)
        ttnn.deallocate(wt)
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
