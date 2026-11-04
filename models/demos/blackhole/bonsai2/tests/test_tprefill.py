# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact ternary prefill linear (tt/tprefill) against an fp32 reference of the same ternary weights."""

import os
import time

import numpy as np
import pytest
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.tprefill.linear import TernaryPrefillLinear, reference


def _weights(K, N, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.integers(-1, 2, size=(K, N)).astype(np.int8)
    s = (rng.random((N, K // 128)) * 0.02 + 0.005).astype(np.float16)
    return t, s


@pytest.mark.parametrize(
    "T, K, N",
    [(64, 1024, 256), (256, 5120, 8192), (128, 5120, 17408), (128, 17408, 5120), (128, 6144, 5120)],
    ids=["small", "k5120n8192", "k5120n17408", "k17408n5120", "k6144n5120"],
)
def test_tprefill(device, T, K, N):
    t, s = _weights(K, N)
    x = torch.randn(T, K).bfloat16()
    lin = TernaryPrefillLinear.from_host(device, t, s)
    xt = ttnn.from_torch(x.reshape(1, 1, T, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y = ttnn.to_torch(lin(xt)).float().reshape(T, N)
    if "DBG_P" in os.environ.get("BONSAI_TPRE_DEFINES", ""):
        s = np.ones_like(s)
    ref = reference(x, t, s)
    bad = ~torch.isfinite(y)
    if bad.any():
        r, c = bad.nonzero()[0].tolist()
        print(
            f"non-finite {bad.float().mean().item():.4f} first at row {r} col {c}; y[0,:4]={y[0, :4].tolist()} ref={ref[0, :4].tolist()}"
        )
    rel = ((y - ref).norm() / ref.norm()).item()
    worst = ((y - ref).abs().max() / ref.abs().max()).item()
    print(f"tprefill T={T} K={K} N={N} rel={rel:.5f} worst={worst:.5f}")
    if rel > 0.01:
        e = (y - ref).reshape(T, N // 32, 32).norm(dim=(0, 2)) / ref.reshape(T, N // 32, 32).norm(dim=(0, 2))
        badt = (e > 0.01).nonzero().flatten().tolist()
        print(f"bad tile columns ({len(badt)}): {badt[:40]}")
    assert rel < 0.01, rel


@pytest.mark.parametrize("K, N", [(5120, 17408), (17408, 5120), (5120, 8192)])
def test_tprefill_time(device, K, N):
    T, R = 2048, 5
    t, s = _weights(K, N)
    lin = TernaryPrefillLinear.from_host(device, t, s)
    xt = ttnn.from_torch(
        torch.randn(1, 1, T, K).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    lin(xt)
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(R):
        lin(xt)
    ttnn.synchronize_device(device)
    dt = (time.perf_counter() - t0) / R
    w = ttnn.from_torch(torch.randn(K, N).bfloat16(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=ttnn.MathFidelity.LoFi)
    ttnn.linear(xt, w, compute_kernel_config=cfg)
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(R):
        ttnn.linear(xt, w, compute_kernel_config=cfg)
    ttnn.synchronize_device(device)
    dd = (time.perf_counter() - t0) / R
    tf = 2 * T * K * N / 1e12
    print(
        f"TPRETIME T={T} K={K} N={N}: ternary {dt * 1e3:.3f} ms ({tf / dt:.0f} TFLOPS), dense bf8 LoFi ttnn.linear {dd * 1e3:.3f} ms ({tf / dd:.0f} TFLOPS)"
    )


@pytest.mark.parametrize("T, K, N", [(128, 5120, 17408), (64, 1024, 512)], ids=["mlp", "small"])
def test_tprefill_glu(device, T, K, N):
    from models.demos.blackhole.bonsai2.tt.ternary.decode_linear3 import pack_glu_records, pack_records_inline

    tg, sg = _weights(K, N, seed=1)
    tu, su = _weights(K, N, seed=2)
    rg, ppr = pack_records_inline(tg, sg)
    ru, _ = pack_records_inline(tu, su)
    rec = ttnn.from_torch(
        torch.from_numpy(pack_glu_records(rg, ru, ppr).view(np.int32)),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    lin = TernaryPrefillLinear(device, rec, ppr, K, 2 * N, glu=True)
    x = torch.randn(T, K).bfloat16()
    xt = ttnn.from_torch(x.reshape(1, 1, T, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y = ttnn.to_torch(lin(xt)).float().reshape(T, N)
    ref = torch.nn.functional.silu(reference(x, tg, sg)) * reference(x, tu, su)
    rel = ((y - ref).norm() / ref.norm()).item()
    print(f"tprefill glu T={T} K={K} N={N} rel={rel:.5f}")
    assert rel < 0.02, rel


@pytest.mark.parametrize("K", [5120, 17408])
def test_prefill_rotation(device, K):
    from models.demos.blackhole.bonsai2.tt.ternary.gguf_weights import fwht_blocks
    from models.demos.blackhole.bonsai2.tt.tprefill.rotate import PrefillRotation

    T = 128
    rng = np.random.default_rng(3)
    signs = rng.choice([-1.0, 1.0], size=K).astype(np.float32)
    x = torch.randn(T, K).bfloat16()
    rot = PrefillRotation(device, signs)
    xt = ttnn.from_torch(x.reshape(1, 1, T, K), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y = ttnn.to_torch(rot(xt)).float().reshape(T, K)
    ref = torch.from_numpy(fwht_blocks(x.float().numpy() * signs, 1024)).float()
    rel = ((y - ref).norm() / ref.norm()).item()
    print(f"prefill rotation K={K} rel={rel:.6f}")
    assert rel < 0.005, rel


def _time(device, fn, R=5):
    fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(R):
        fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / R * 1e3


def test_tprefill_parts_time(device):
    """The exact prefill MLP's pieces at T=2048: rotations, gate/up GLU, down."""
    from models.demos.blackhole.bonsai2.tt.ternary.decode_linear3 import pack_glu_records, pack_records_inline
    from models.demos.blackhole.bonsai2.tt.tprefill.rotate import PrefillRotation

    T = 2048
    out = []
    for K in (5120, 17408):
        rot = PrefillRotation(device, np.ones(K, np.float32))
        xt = ttnn.from_torch(
            torch.randn(1, T, K).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        out.append(f"rot{K}={_time(device, lambda: rot(xt)):.3f}")
    K, N = 5120, 17408
    tg, sg = _weights(K, N, seed=1)
    rg, ppr = pack_records_inline(tg, sg)
    rec = ttnn.from_torch(
        torch.from_numpy(pack_glu_records(rg, rg, ppr).view(np.int32)),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    glu = TernaryPrefillLinear(device, rec, ppr, K, 2 * N, glu=True)
    xt = ttnn.from_torch(torch.randn(1, T, K).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out.append(f"gateup_glu={_time(device, lambda: glu(xt)):.3f}")
    t, s = _weights(N, K)
    down = TernaryPrefillLinear.from_host(device, t, s)
    ht = ttnn.from_torch(torch.randn(1, T, N).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out.append(f"down={_time(device, lambda: down(ht)):.3f}")
    print("PARTS_MS " + " ".join(out))
