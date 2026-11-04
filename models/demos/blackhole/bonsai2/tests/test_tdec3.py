# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.ternary.decode_linear3 import TernaryDecodeLinear3, reference


@pytest.mark.parametrize("sfpu_iter", [10])
@pytest.mark.parametrize("reinit", [True, False])
@pytest.mark.parametrize("scale_fp32", [False])
@pytest.mark.parametrize("K, N", [(1024, 64), (5120, 1024), (5120, 17408)])
def test_tdec3(device, K, N, scale_fp32, sfpu_iter, reinit):
    rng = np.random.default_rng(0)
    t = rng.integers(-1, 2, size=(K, N)).astype(np.int8)
    s = (rng.random((N, K // 128)) * 0.02 + 0.005).astype(np.float16)
    x = torch.randn(K).bfloat16()
    lin = TernaryDecodeLinear3.from_host(device, t, s, scale_fp32=scale_fp32)
    act = ttnn.from_torch(
        x.reshape(1, -1).repeat(8, 1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    y = ttnn.to_torch(lin(act, sfpu_iter=sfpu_iter, sfpu_reinit=reinit)).float()[0]
    ref = reference(x, t, s)
    rel = ((y - ref).norm() / ref.norm()).item()
    print(
        f"reinit={reinit} iter={sfpu_iter} fp32={scale_fp32} K={K} N={N} rel={rel:.5f} y[:4]={y[:4].tolist()} ref[:4]={ref[:4].tolist()}"
    )
    err = ((y - ref).abs() / ref.abs().clamp(min=1e-3))[:64]
    print("bad cols", [i for i in range(min(64, N)) if err[i] > 0.05])
    print("ratio y/ref cols 14..20", (y[14:21] / ref[14:21]).tolist())
    for _ in range(20):
        lin(act, sfpu_iter=sfpu_iter, sfpu_reinit=reinit)
    ttnn.synchronize_device(device)
    assert rel < 0.02, rel


@pytest.mark.parametrize("variant", ["SKIP_SFPU", "SKIP_MM", "SKIP_BOTH"])
def test_tdec3_parts(device, variant):
    K, N = 5120, 17408
    rng = np.random.default_rng(0)
    t = rng.integers(-1, 2, size=(K, N)).astype(np.int8)
    s = (rng.random((N, K // 128)) * 0.02 + 0.005).astype(np.float16)
    lin = TernaryDecodeLinear3.from_host(device, t, s)
    act = ttnn.from_torch(
        torch.randn(1, K).bfloat16().repeat(8, 1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    d = [("SKIP_SFPU", "1"), ("SKIP_MM", "1")] if variant == "SKIP_BOTH" else [(variant, "1")]
    for _ in range(21):
        lin(act, extra_defines=d)
    ttnn.synchronize_device(device)
    print(f"reinit=parts {variant}")


@pytest.mark.parametrize("K, N", [(1024, 64), (5120, 17408)])
def test_tdec3_g16(device, K, N):
    rng = np.random.default_rng(0)
    t = rng.integers(-1, 2, size=(K, N)).astype(np.int8)
    s = (rng.random((N, K // 128)) * 0.02 + 0.005).astype(np.float16)
    x = torch.randn(K).bfloat16()
    lin = TernaryDecodeLinear3.from_host(device, t, s, group16=True)
    act = ttnn.from_torch(
        x.reshape(1, -1).repeat(8, 1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    y = ttnn.to_torch(lin(act)).float()[0]
    ref = reference(x, t, s)
    rel = ((y - ref).norm() / ref.norm()).item()
    print(f"reinit=g16 K={K} N={N} rel={rel:.5f}")
    for _ in range(20):
        lin(act)
    ttnn.synchronize_device(device)
    assert rel < 0.02, rel


@pytest.mark.parametrize("variant", ["fFULL", "fNOSFPU", "fNOMM", "fNONE"])
def test_tdec3_fake(device, variant):
    """Compute cost alone: the readers push pages without reading DRAM."""
    K, N = 5120, 17408
    rng = np.random.default_rng(0)
    t = rng.integers(-1, 2, size=(K, N)).astype(np.int8)
    s = (rng.random((N, K // 128)) * 0.02 + 0.005).astype(np.float16)
    lin = TernaryDecodeLinear3.from_host(device, t, s, group16=True)
    act = ttnn.from_torch(
        torch.randn(1, K).bfloat16().repeat(8, 1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    d = {
        "fFULL": [],
        "fNOSFPU": [("SKIP_SFPU", "1")],
        "fNOMM": [("SKIP_MM", "1")],
        "fNONE": [("SKIP_SFPU", "1"), ("SKIP_MM", "1")],
    }[variant]
    for _ in range(21):
        lin(act, extra_defines=d, dm_defines=[("FAKE_STREAM", "1")])
    ttnn.synchronize_device(device)


@pytest.mark.parametrize("K, N", [(1024, 64), (6144, 5120), (5120, 17408)])
def test_tdec3_inl(device, K, N):
    rng = np.random.default_rng(0)
    t = rng.integers(-1, 2, size=(K, N)).astype(np.int8)
    s = (rng.random((N, K // 128)) * 0.02 + 0.005).astype(np.float16)
    x = torch.randn(K).bfloat16()
    lin = TernaryDecodeLinear3.from_host(device, t, s, inline_scales=True)
    act = ttnn.from_torch(
        x.reshape(1, -1).repeat(8, 1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    y = ttnn.to_torch(lin(act)).float()[0]
    ref = reference(x, t, s)
    rel = ((y - ref).norm() / ref.norm()).item()
    print(f"reinit=inl K={K} N={N} rel={rel:.5f}")
    for _ in range(20):
        lin(act)
    ttnn.synchronize_device(device)
    assert rel < 0.02, rel
