# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Prefill MoE combine (tt/fused/pf_combine.py) vs the op chain of prefill_model._experts_unified (embedding of the
back rows, multiply by the bf16 weights, sum over k): as close to the exactly rounded sum as the op chain, at
T = 256 and 8192, plus the time of both."""

import time

import pytest
import torch

import ttnn
from models.demos.inkling2_mini.tt.fused.pf_combine import PfCombine

K, C = 8, 2304


def timed(dev, fn, reps=3):
    fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / reps * 1e3


@pytest.mark.parametrize("t", [256, 8192])
def test_pf_combine(device, t):
    dev = device
    g = torch.Generator().manual_seed(t)
    rows = t * K + 4096
    rm = ttnn.ROW_MAJOR_LAYOUT
    y = ttnn.from_torch(torch.randn(rows, C, generator=g), dtype=ttnn.bfloat16, layout=rm, device=dev)
    back = ttnn.from_torch(torch.randint(0, rows, (1, K * t), generator=g, dtype=torch.int32), dtype=ttnn.uint32,
                           layout=rm, device=dev)  # fmt: skip
    wk = ttnn.from_torch(torch.rand(1, K, t, 1, generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                         device=dev)  # fmt: skip
    mod = PfCombine(dev)

    def ref():
        yb = ttnn.reshape(ttnn.embedding(back, y, layout=ttnn.TILE_LAYOUT), [1, K, t, C])
        return ttnn.sum(ttnn.multiply(yb, wk), dim=1, keepdim=True)

    a, b = ttnn.to_torch(ref()).float(), ttnn.to_torch(mod(y, back, wk)).float()
    # Not bit-equal: the op chain and the fused kernel round the products in different dest formats. Both must be
    # as close to the exactly rounded sum (fp32 sum of the exact products, then bf16).
    yt = ttnn.to_torch(y).float()[ttnn.to_torch(back).long().view(K, t)]  # [K, t, C]
    prod = yt * ttnn.to_torch(wk).float().view(K, t, 1)
    exact = prod.sum(0).bfloat16().float()
    mag = prod.abs().sum(0)  # rounding errors scale with the terms, not with a cancelled sum
    bad_ops, bad_fused = int((a[0, 0] != exact).sum()), int((b[0, 0] != exact).sum())
    rel = float(((a[0, 0] - b[0, 0]).abs() / mag.clamp(min=1e-6)).max())
    print(
        f"PF_COMBINE t={t} equal={torch.equal(a, b)} off_exact ops={bad_ops} fused={bad_fused} max_rel_diff={rel:.3g}"
    )
    w_tk = ttnn.permute(wk, (0, 3, 2, 1))  # the router's layout [1, 1, t, K]: the reader builds the weight tiles
    assert torch.equal(b, ttnn.to_torch(mod(y, back, w_tk)).float())
    narrow = PfCombine(dev, cw=8)  # same sums, other units: bit-equal
    assert torch.equal(b, ttnn.to_torch(narrow(y, back, wk)).float())
    if t == 8192:
        print(f"PF_COMBINE_MS t={t} ops {timed(dev, ref):.2f} fused {timed(dev, lambda: mod(y, back, wk)):.2f}"
              f" cw8 {timed(dev, lambda: narrow(y, back, wk)):.2f}")  # fmt: skip
    assert bad_fused <= 1.05 * bad_ops + 16 and rel < 2**-6, (bad_ops, bad_fused, rel)
