# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ternary decode linear on the board against an fp32 torch reference (random ternary weights, fp16 scales)."""

import time

import numpy as np
import pytest
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.ternary.decode_linear import TernaryDecodeLinear, reference


@pytest.mark.parametrize("debug", [0, 8, 9])
@pytest.mark.parametrize("K, N", [(1024, 64), (5120, 1024), (5120, 17408)])
def test_ternary_decode_linear(device, K, N, debug):
    rng = np.random.default_rng(0)
    t = rng.integers(-1, 2, size=(K, N)).astype(np.int8)
    s = (rng.random((N, K // 128)) * 0.02 + 0.005).astype(np.float16)
    x = torch.randn(K).bfloat16()
    lin = TernaryDecodeLinear.from_host(device, t, s)
    act = ttnn.from_torch(x.reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    y = ttnn.to_torch(lin(act, debug=debug)).float()[0]
    ref = reference(x, t, s)
    rel = ((y - ref).norm() / ref.norm()).item()
    pcc = torch.corrcoef(torch.stack([y, ref]))[0, 1].item()
    print(f"K={K} N={N} rel={rel:.5f} pcc={pcc:.6f} y[:4]={y[:4].tolist()} ref[:4]={ref[:4].tolist()}")
    for _ in range(3):
        lin(act, debug=debug)
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    n = 20
    for _ in range(n):
        lin(act, debug=debug)
    ttnn.synchronize_device(device)
    us = (time.perf_counter() - t0) / n * 1e6
    mb = lin.w.shape[0] * lin.w.shape[1] * 4 / 1e6
    print(f"K={K} N={N} host-timed {us:.1f} us/call, weight bytes {mb:.2f} MB -> {mb / us * 1e3:.1f} GB/s")
    assert debug or rel < 0.02, rel
