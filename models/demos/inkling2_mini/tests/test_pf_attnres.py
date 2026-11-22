# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Prefill multi-row attn-res (tt/fused/pf_attnres.py) vs prefill_model._AttnRes: outputs bit-equal over depths
1..6 sealed states and 1 or 3 queries, T = 256 and 2048, plus the time of both at T = 2048."""

import time

import pytest
import torch

import ttnn
from models.demos.inkling2_mini.tt.fused.pf_attnres import PfAttnRes
from models.demos.inkling2_mini.tt.prefill_model import EPS, _AttnRes


def timed(dev, fn, reps=3):
    fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / reps * 1e3


@pytest.mark.parametrize("t,depth,nq", [(256, 1, 1), (256, 3, 3), (256, 6, 3), (2048, 4, 1), (2048, 5, 3)])
def test_pf_attnres(device, t, depth, nq):
    dev = device
    g = torch.Generator().manual_seed(t + depth + nq)
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )  # fmt: skip
    q = torch.randn(32, 2304, generator=g)
    put = lambda x: ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
    states = [put(torch.randn(1, 1, t, 2304, generator=g) * (1 + i)) for i in range(depth)]
    normed = [ttnn.rms_norm(s, epsilon=EPS, compute_kernel_config=ckc) for s in states]
    partial = put(torch.randn(1, 1, t, 2304, generator=g) * 3)
    ref_mod, new_mod = _AttnRes(dev, q, nq, ckc), PfAttnRes(dev, q, nq, ckc)
    ref = [ttnn.to_torch(x) for x in ref_mod(normed, partial)]
    new = [ttnn.to_torch(x) for x in new_mod(normed, partial)]
    for i, (a, b) in enumerate(zip(ref, new)):
        diff = float((a - b).abs().max())
        print(f"PF_ATTNRES t={t} depth={depth} nq={nq} q={i} equal={torch.equal(a, b)} maxdiff={diff:.3g}")
        assert torch.equal(a, b), diff
    if t == 2048:
        ms_ref = timed(dev, lambda: ref_mod(normed, partial))
        ms_new = timed(dev, lambda: new_mod(normed, partial))
        print(f"PF_ATTNRES_MS t={t} depth={depth} nq={nq} ops {ms_ref:.2f} fused {ms_new:.2f}")
