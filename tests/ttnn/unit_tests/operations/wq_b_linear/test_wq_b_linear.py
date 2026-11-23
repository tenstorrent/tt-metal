# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Acceptance test for wq_b_linear: y = x @ W, x [1,1,160,1536] bf16, W [1536,4096] bf8_b.

The op has exactly one problem shape, so the parametrization covers seeds and both
call patterns instead of shapes. Precision gate: PCC >= helpers.TOLERANCES
(0.9998) against an fp64 matmul of x and the device-quantized W.

Do not modify: this file is the spec the implementation must pass.
"""

import pytest
import torch
import ttnn

from ttnn.operations.wq_b_linear import wq_b_linear

from eval.golden_tests.wq_b_linear.helpers import TOLERANCES, WEIGHT_SCALE

X_SHAPE = (1, 1, 160, 1536)
W_SHAPE = (1536, 4096)
Y_SHAPE = [1, 1, 160, 4096]
PCC_MIN = TOLERANCES[(ttnn.bfloat8_b, False)]


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def _make_inputs(device, seed):
    torch.manual_seed(seed)
    x = torch.randn(X_SHAPE, dtype=torch.float32).to(torch.bfloat16)
    w = (torch.randn(W_SHAPE, dtype=torch.float32) * WEIGHT_SCALE).to(torch.bfloat16)
    tx = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tw = ttnn.from_torch(
        w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return x, tx, tw


def _reference(x, tw):
    # The weight as the device holds it (after bfloat8_b block quantization).
    w_q = ttnn.to_torch(tw)
    return torch.matmul(x.to(torch.float64), w_q.to(torch.float64))


def _check_output(out, expected):
    assert list(out.shape) == Y_SHAPE, f"shape {list(out.shape)}"
    assert out.dtype == ttnn.bfloat16, f"dtype {out.dtype}"
    assert out.layout == ttnn.TILE_LAYOUT, f"layout {out.layout}"
    mc = out.memory_config()
    assert mc.buffer_type == ttnn.BufferType.L1, f"buffer type {mc.buffer_type}"
    assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, f"memory layout {mc.memory_layout}"
    got = ttnn.to_torch(out)
    pcc = _pcc(got, expected)
    assert pcc >= PCC_MIN, f"PCC {pcc:.6f} < {PCC_MIN}"
    return got


@pytest.mark.parametrize("seed", [42, 0, 7])
def test_wq_b_linear_default_config(device, seed):
    x, tx, tw = _make_inputs(device, seed)
    out = wq_b_linear(tx, tw)
    _check_output(out, _reference(x, tw))


@pytest.mark.parametrize("seed", [42])
def test_wq_b_linear_explicit_config(device, seed):
    x, tx, tw = _make_inputs(device, seed)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False)
    out = wq_b_linear(tx, tw, compute_kernel_config=cfg)
    _check_output(out, _reference(x, tw))


def test_wq_b_linear_repeat_bit_identical(device):
    x, tx, tw = _make_inputs(device, 42)
    first = _check_output(wq_b_linear(tx, tw), _reference(x, tw))
    for _ in range(3):
        again = ttnn.to_torch(wq_b_linear(tx, tw))
        assert torch.equal(first, again), "output differs between repeated calls"


def test_wq_b_linear_default_compute_kernel_config():
    from ttnn.operations.wq_b_linear import default_compute_kernel_config

    cfg = default_compute_kernel_config()
    assert cfg.math_fidelity == ttnn.MathFidelity.HiFi2
    assert cfg.fp32_dest_acc_en is False
    assert cfg.math_approx_mode is False


def test_wq_b_linear_rejects_wrong_x_shape(device, expect_error):
    torch.manual_seed(42)
    x = torch.randn((1, 1, 128, 1536), dtype=torch.bfloat16)
    w = torch.randn(W_SHAPE, dtype=torch.bfloat16) * WEIGHT_SCALE
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    # Host-side contract check; the message text is not part of the spec.
    with expect_error((ValueError, RuntimeError), ""):
        wq_b_linear(tx, tw)


def test_wq_b_linear_rejects_wrong_w_shape(device, expect_error):
    torch.manual_seed(42)
    x = torch.randn(X_SHAPE, dtype=torch.bfloat16)
    w = torch.randn((1536, 2048), dtype=torch.bfloat16) * WEIGHT_SCALE
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    # Host-side contract check; the message text is not part of the spec.
    with expect_error((ValueError, RuntimeError), ""):
        wq_b_linear(tx, tw)


@pytest.mark.parametrize("which", ["x", "w"])
def test_wq_b_linear_rejects_non_dram_input(device, expect_error, which):
    torch.manual_seed(42)
    x = torch.randn(X_SHAPE, dtype=torch.bfloat16)
    w = torch.randn(W_SHAPE, dtype=torch.bfloat16) * WEIGHT_SCALE
    x_mc = ttnn.L1_MEMORY_CONFIG if which == "x" else ttnn.DRAM_MEMORY_CONFIG
    w_mc = ttnn.L1_MEMORY_CONFIG if which == "w" else ttnn.DRAM_MEMORY_CONFIG
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=x_mc)
    tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=w_mc)
    # Host-side contract check; the message text is not part of the spec.
    with expect_error((ValueError, RuntimeError), ""):
        wq_b_linear(tx, tw)
