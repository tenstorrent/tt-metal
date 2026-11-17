# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Acceptance test for rms_norm (Phase 0: fp32_dest_acc_en=True, interleaved)."""
import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm

PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995}
TORCH_DT = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}


def torch_rms_norm(x, gamma, eps):
    x = x.float()
    y = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    if gamma is not None:
        y = y * gamma.float().reshape(-1)
    return y


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def cfg():
    return ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)


SHAPES = [
    (1, 1, 32, 32),      # single tile
    (1, 1, 64, 128),     # multi tile
    (2, 3, 64, 96),      # multi batch, non-square
    (4, 1, 32, 256),
    (1, 1, 32, 50),      # W non-aligned
    (1, 1, 17, 64),      # H non-aligned
    (2, 1, 100, 47),     # both non-aligned
    (1, 32, 128),        # 3D
    (128, 100),          # 2D, W non-aligned
    (1, 1, 32, 4096),    # wide
]


def run(device, shape, dtype, layout, use_gamma, eps=1e-6, gamma_dtype=None):
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.float32).to(TORCH_DT[dtype])
    W = shape[-1]
    g = torch.randn((1, 1, 1, W), dtype=torch.float32) if use_gamma else None
    tt_x = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device)
    tt_g = None
    if use_gamma:
        gd = gamma_dtype or dtype
        tt_g = ttnn.from_torch(g.to(TORCH_DT[gd]), dtype=gd, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out = rms_norm(tt_x, gamma=tt_g, epsilon=eps, compute_kernel_config=cfg())
    assert list(out.shape) == list(shape)
    assert out.layout == layout
    res = ttnn.to_torch(out).float()
    ref = torch_rms_norm(x, g.to(TORCH_DT[gamma_dtype or dtype]) if use_gamma else None, eps)
    assert pcc(res, ref) >= PCC[dtype], f"PCC {pcc(res, ref)}"


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("use_gamma", [False, True], ids=["no_gamma", "gamma"])
def test_rms_norm(device, shape, dtype, layout, use_gamma):
    run(device, shape, dtype, layout, use_gamma)


@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_rms_norm_epsilon(device, eps):
    run(device, (1, 1, 64, 128), ttnn.bfloat16, ttnn.TILE_LAYOUT, True, eps=eps)


def test_rms_norm_mixed_gamma_dtype(device):
    run(device, (1, 1, 64, 128), ttnn.bfloat16, ttnn.TILE_LAYOUT, True, gamma_dtype=ttnn.float32)


def test_rms_norm_default_config(device):
    x = torch.randn((1, 1, 32, 64), dtype=torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(rms_norm(tt_x)).float()
    assert pcc(out, torch_rms_norm(x, None, 1e-6)) >= PCC[ttnn.bfloat16]


def test_rms_norm_padding_not_in_mean(device):
    # Small non-aligned W: a padded-width mean would be off by >15%.
    torch.manual_seed(42)
    x = torch.randn((1, 1, 32, 72), dtype=torch.float32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    res = ttnn.to_torch(rms_norm(tt_x, compute_kernel_config=cfg())).float()
    assert torch.allclose(res, torch_rms_norm(x, None, 1e-6), rtol=2e-2, atol=2e-2)


def test_rms_norm_validation(device):
    # rank < 2
    t1 = ttnn.from_torch(torch.randn((1, 1, 32, 64)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    t1 = ttnn.reshape(t1, (1, 1, 32, 64))
    with pytest.raises((ValueError, RuntimeError)):
        rms_norm(ttnn.from_torch(torch.randn(64), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device))
    # gamma width mismatch
    g = ttnn.from_torch(torch.randn((1, 1, 1, 32)), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with pytest.raises((ValueError, RuntimeError)):
        rms_norm(t1, gamma=g)
