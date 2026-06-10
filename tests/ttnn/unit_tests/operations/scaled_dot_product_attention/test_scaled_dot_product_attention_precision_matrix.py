# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision matrix for scaled_dot_product_attention (Refinement 1).

Single authoritative precision characterization: shapes x dtype x
math_fidelity x fp32_dest_acc_en x distribution. All metrics printed for
every case; assertion is on PCC only.

Skips (documented hardware / structural limits, enforced by the entry point):
- HiFi4 + fp32_dest_acc_en + bf16/bf8b inputs: known-bad on Wormhole B0
  (issue #38306, matmul_block_helpers.hpp) — entry point raises ValueError.
- fp32/bf8b inputs + fp32_dest_acc_en=False: 16-bit DEST format pairing is
  structurally unsupported (probe_008: fp32 pcc 0.008, bf8b NaN) — entry
  point raises NotImplementedError.

Results file: precision_matrix_results.md (same directory).
"""

import math

import pytest
import torch
import ttnn

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

# Non-aligned shapes are outside SUPPORTED["alignment"] (Refinement 4 scope),
# so the shape axis is tile-aligned only: (B, H, S, D), small -> large.
SHAPES = [
    pytest.param((1, 1, 32, 32), id="1x1x32x32_small"),
    pytest.param((1, 1, 128, 64), id="1x1x128x64"),
    pytest.param((1, 1, 64, 32), id="1x1x64x32"),
    pytest.param((1, 4, 256, 64), id="1x4x256x64"),
    pytest.param((2, 4, 128, 64), id="2x4x128x64_batch"),
    pytest.param((1, 1, 128, 256), id="1x1x128x256_wide_d"),
    pytest.param((1, 1, 2048, 64), id="1x1x2048x64_long"),
    pytest.param((1, 8, 512, 128), id="1x8x512x128_large"),
]

PCC_FLOOR = {ttnn.float32: 0.999, ttnn.bfloat16: 0.99, ttnn.bfloat8_b: 0.99}


@pytest.mark.parametrize("distribution", [pytest.param("rand", id="uniform"), pytest.param("randn", id="normal")])
@pytest.mark.parametrize("fp32_acc", [pytest.param(True, id="fp32_acc"), pytest.param(False, id="bf16_acc")])
@pytest.mark.parametrize(
    "math_fidelity",
    [
        pytest.param(ttnn.MathFidelity.HiFi4, id="HiFi4"),
        pytest.param(ttnn.MathFidelity.HiFi3, id="HiFi3"),
        pytest.param(ttnn.MathFidelity.HiFi2, id="HiFi2"),
        pytest.param(ttnn.MathFidelity.LoFi, id="LoFi"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(ttnn.bfloat16, id="bf16"),
        pytest.param(ttnn.float32, id="fp32"),
        pytest.param(ttnn.bfloat8_b, id="bfp8"),
    ],
)
@pytest.mark.parametrize("shape", SHAPES)
def test_scaled_dot_product_attention_precision_matrix(device, shape, dtype, math_fidelity, fp32_acc, distribution):
    if math_fidelity == ttnn.MathFidelity.HiFi4 and fp32_acc and dtype != ttnn.float32:
        pytest.skip("HiFi4 + fp32 DEST + bf16-family inputs known-bad on Wormhole B0 (issue #38306)")
    if not fp32_acc and dtype != ttnn.bfloat16:
        pytest.skip("fp32/bf8b inputs require fp32_dest_acc_en (16-bit DEST pairing structurally unsupported)")

    B, H, S, D = shape
    torch.manual_seed(0)
    gen = torch.rand if distribution == "rand" else torch.randn
    q, k, v = gen(B, H, S, D), gen(B, H, S, D), gen(B, H, S, D)

    scale = 1.0 / math.sqrt(D)
    expected = torch.softmax(q @ k.transpose(-2, -1) * scale, -1) @ v

    config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_acc,
        math_approx_mode=False,
    )
    t = lambda x: ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(scaled_dot_product_attention(t(q), t(k), t(v), compute_kernel_config=config)).float()

    g, a = expected.flatten().float(), got.flatten()
    pcc = torch.corrcoef(torch.stack([g, a]))[0, 1].item()
    abs_err = (a - g).abs()
    max_abs = abs_err.max().item()
    median_abs = abs_err.median().item()
    p99_abs = torch.quantile(abs_err, 0.99).item()
    rel_rms = (abs_err.pow(2).mean().sqrt() / g.pow(2).mean().sqrt().clamp(min=1e-10)).item()
    print(
        f"PRECISION shape={shape} dtype={dtype} fid={math_fidelity} fp32_acc={fp32_acc} "
        f"dist={distribution} pcc={pcc:.6f} max_abs={max_abs:.5f} median_abs={median_abs:.6f} "
        f"p99_abs={p99_abs:.5f} rel_rms={rel_rms:.5f}"
    )
    assert pcc >= PCC_FLOOR[dtype], f"pcc {pcc:.6f} < {PCC_FLOOR[dtype]}"


# ---------------------------------------------------------------------------
# compute_kernel_config exposure + guard behavior
# ---------------------------------------------------------------------------


def _tensors(device, dtype=ttnn.bfloat16):
    torch.manual_seed(0)
    t = lambda: ttnn.from_torch(torch.randn(1, 1, 64, 64), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    return t(), t(), t()


def test_default_config_matches_explicit_hifi2(device):
    """Passing nothing == passing the documented defaults (HiFi2 + fp32 dest)."""
    q, k, v = _tensors(device)
    out_default = ttnn.to_torch(scaled_dot_product_attention(q, k, v))
    config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
        dst_full_sync_en=False,
    )
    out_explicit = ttnn.to_torch(scaled_dot_product_attention(q, k, v, compute_kernel_config=config))
    assert torch.equal(out_default, out_explicit)


def test_hifi4_fp32_dest_bf16_rejected(device):
    q, k, v = _tensors(device, ttnn.bfloat16)
    config = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    with pytest.raises(ValueError, match="38306"):
        scaled_dot_product_attention(q, k, v, compute_kernel_config=config)


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat8_b], ids=["fp32", "bfp8"])
def test_bf16_dest_non_bf16_rejected(device, dtype):
    q, k, v = _tensors(device, dtype)
    config = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False)
    with pytest.raises(NotImplementedError, match="fp32_dest_acc_en"):
        scaled_dot_product_attention(q, k, v, compute_kernel_config=config)


def test_dst_full_sync_en(device):
    q, k, v = _tensors(device)
    config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, dst_full_sync_en=True
    )
    out = ttnn.to_torch(scaled_dot_product_attention(q, k, v, compute_kernel_config=config))
    assert torch.isfinite(out).all()
