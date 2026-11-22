# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Prefill softcapped SwiGLU (tt/fused/pf_swiglu.py) vs the op chain of prefill_model._experts_combine (column
slices + moe.softcapped_swiglu): bit-equal on the grouped [1, G, 64, 2304] gate/up shape, plus the time of both."""

import time

import pytest
import torch

import ttnn
from models.demos.inkling2_mini.tt.fused.pf_swiglu import PfSwiglu
from models.demos.inkling2_mini.tt.moe import INTER, softcapped_swiglu


def timed(dev, fn, reps=3):
    fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / reps * 1e3


@pytest.mark.parametrize("g", [8, 1152])
def test_pf_swiglu(device, g):
    dev = device
    x = torch.randn(1, g, 64, 2 * INTER, generator=torch.Generator().manual_seed(g)) * 6
    gu = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    mod = PfSwiglu(dev)

    def ref():
        gg = ttnn.slice(gu, [0, 0, 0, 0], [1, g, 64, INTER])
        uu = ttnn.slice(gu, [0, 0, 0, INTER], [1, g, 64, 2 * INTER])
        return softcapped_swiglu(gg, uu)

    a, b = ttnn.to_torch(ref()).float(), ttnn.to_torch(mod(gu)).float()
    diff = float((a - b).abs().max())
    print(f"PF_SWIGLU g={g} equal={torch.equal(a, b)} maxdiff={diff:.3g} nbad={int((a != b).sum())}")
    if g == 1152:
        print(f"PF_SWIGLU_MS g={g} ops {timed(dev, ref):.2f} fused {timed(dev, lambda: mod(gu)):.2f}")
    assert torch.equal(a, b), diff
