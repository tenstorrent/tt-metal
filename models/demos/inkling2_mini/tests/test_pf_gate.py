# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Prefill message-gate tail (tt/fused/pf_gate.py) vs the tail of prefill_model._Gate: output bit-equal at
T = 256 and 2048, plus the time of both at T = 2048."""

import time

import pytest
import torch

import ttnn
from models.demos.inkling2_mini.tt.fused.pf_gate import PfGateTail


def ref_tail(z, forget, gain, h, m, ckc):  # prefill_model._Gate.__call__ after the down_proj
    z = ttnn.typecast(z, ttnn.float32)
    f = ttnn.tanh(ttnn.matmul(z, forget, dtype=ttnn.float32, compute_kernel_config=ckc))
    g = ttnn.tanh(ttnn.matmul(z, gain, dtype=ttnn.float32, compute_kernel_config=ckc))
    return ttnn.addcmul(ttnn.addcmul(h, f, h, value=-1.0), g, m)


def timed(dev, fn, reps=3):
    fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / reps * 1e3


@pytest.mark.parametrize("t", [256, 2048])
def test_pf_gate(device, t):
    dev = device
    g = torch.Generator().manual_seed(t)
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )  # fmt: skip
    put = lambda a, dt: ttnn.from_torch(a, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
    z = put(torch.randn(1, 1, t, 64, generator=g) * 2, ttnn.bfloat16)
    fw = put(torch.randn(1, 1, 64, 2304, generator=g) * 0.2, ttnn.float32)
    gw = put(torch.randn(1, 1, 64, 2304, generator=g) * 0.2, ttnn.float32)
    h = put(torch.randn(1, 1, t, 2304, generator=g) * 3, ttnn.float32)
    m = put(torch.randn(1, 1, t, 2304, generator=g), ttnn.float32)
    mod = PfGateTail(dev)
    a = ttnn.to_torch(ref_tail(z, fw, gw, h, m, ckc))
    b = ttnn.to_torch(mod(z, fw, gw, h, m))
    diff = float((a - b).abs().max())
    print(f"PF_GATE t={t} equal={torch.equal(a, b)} maxdiff={diff:.3g} nbad={int((a != b).sum())}")
    assert torch.equal(a, b), diff
    if t == 2048:
        ms_ref = timed(dev, lambda: ref_tail(z, fw, gw, h, m, ckc))
        ms_new = timed(dev, lambda: mod(z, fw, gw, h, m))
        print(f"PF_GATE_MS t={t} ops {ms_ref:.2f} fused {ms_new:.2f}")
