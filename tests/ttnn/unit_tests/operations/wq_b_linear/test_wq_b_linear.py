# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Acceptance test for wq_b_linear: y = x @ W on the single DSA-indexer wq_b shape.

Immutable spec for the implementer. The reference is an fp64 matmul of x and the
device-quantized (bfloat8_b) W; the gate is the golden suite's PCC limit.
"""

import pytest
import torch
import ttnn

from eval.golden_tests.wq_b_linear.helpers import TOLERANCES, WEIGHT_SCALE
from eval.metrics import Tolerance, check_output
from ttnn.operations.wq_b_linear import default_compute_kernel_config, wq_b_linear

X_SHAPE = (1, 1, 160, 1536)
W_SHAPE = (1536, 4096)
Y_SHAPE = [1, 1, 160, 4096]
PCC = TOLERANCES[(ttnn.bfloat8_b, False)]


def torch_wq_b_linear(x, w):
    return torch.matmul(x.to(torch.float64), w.to(torch.float64))


def _make_inputs(case):
    torch.manual_seed(42)
    x = torch.randn(X_SHAPE, dtype=torch.float32)
    w = torch.randn(W_SHAPE, dtype=torch.float32) * WEIGHT_SCALE
    if case == "x_scaled":
        x = x * 8.0
    elif case == "w_scaled":
        w = w * 10.0
    elif case == "x_offset":
        x = x + 0.5
    return x.to(torch.bfloat16), w.to(torch.bfloat16)


def _to_device(
    x,
    w,
    device,
    x_mem=ttnn.DRAM_MEMORY_CONFIG,
    w_mem=ttnn.DRAM_MEMORY_CONFIG,
    x_dtype=ttnn.bfloat16,
    w_dtype=ttnn.bfloat8_b,
):
    tx = ttnn.from_torch(x, dtype=x_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=x_mem)
    tw = ttnn.from_torch(w, dtype=w_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=w_mem)
    return tx, tw


def _assert_l1_interleaved(y):
    mc = y.memory_config()
    assert mc.buffer_type == ttnn.BufferType.L1
    assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED


CONFIGS = {
    "default": None,
    "explicit_hifi2": "explicit",
}


@pytest.mark.parametrize("config_id", list(CONFIGS))
@pytest.mark.parametrize("case", ["standard", "x_scaled", "w_scaled", "x_offset"])
def test_wq_b_linear(device, case, config_id):
    x, w = _make_inputs(case)
    tx, tw = _to_device(x, w, device)
    expected = torch_wq_b_linear(x, ttnn.to_torch(tw))

    if CONFIGS[config_id] is None:
        y = wq_b_linear(tx, tw)
    else:
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False)
        y = wq_b_linear(tx, tw, compute_kernel_config=cfg)

    _assert_l1_interleaved(y)
    check_output(
        y,
        expected,
        shape=Y_SHAPE,
        dtype=ttnn.bfloat16,
        expected_layout=ttnn.TILE_LAYOUT,
        tolerance=Tolerance(pcc=PCC),
    )


def test_default_compute_kernel_config():
    cfg = default_compute_kernel_config()
    assert cfg.math_fidelity == ttnn.MathFidelity.HiFi2
    assert cfg.fp32_dest_acc_en is False
    assert cfg.math_approx_mode is False


def test_repeated_calls_bit_identical(device):
    x, w = _make_inputs("standard")
    tx, tw = _to_device(x, w, device)
    first = ttnn.to_torch(wq_b_linear(tx, tw))
    for _ in range(3):
        assert torch.equal(ttnn.to_torch(wq_b_linear(tx, tw)), first)


@pytest.mark.parametrize(
    "x_shape,w_shape",
    [
        ((1, 1, 128, 1536), W_SHAPE),
        ((1, 1, 160, 1024), (1024, 4096)),
        (X_SHAPE, (1536, 2048)),
    ],
    ids=["x_rows", "k_dim", "w_cols"],
)
def test_rejects_wrong_shape(device, expect_error, x_shape, w_shape):
    torch.manual_seed(42)
    x = torch.randn(x_shape).to(torch.bfloat16)
    w = (torch.randn(w_shape) * WEIGHT_SCALE).to(torch.bfloat16)
    tx, tw = _to_device(x, w, device)
    with expect_error((ValueError, RuntimeError), ""):
        wq_b_linear(tx, tw)


@pytest.mark.parametrize("which", ["x_in_l1", "w_in_l1"])
def test_rejects_non_dram_inputs(device, expect_error, which):
    x, w = _make_inputs("standard")
    x_mem = ttnn.L1_MEMORY_CONFIG if which == "x_in_l1" else ttnn.DRAM_MEMORY_CONFIG
    w_mem = ttnn.L1_MEMORY_CONFIG if which == "w_in_l1" else ttnn.DRAM_MEMORY_CONFIG
    tx, tw = _to_device(x, w, device, x_mem=x_mem, w_mem=w_mem)
    with expect_error((ValueError, RuntimeError), ""):
        wq_b_linear(tx, tw)


@pytest.mark.parametrize("which", ["x_bf8b", "w_bf16", "fp32_dest_acc"])
def test_rejects_unsupported_cells(device, expect_error, which):
    x, w = _make_inputs("standard")
    x_dtype = ttnn.bfloat8_b if which == "x_bf8b" else ttnn.bfloat16
    w_dtype = ttnn.bfloat16 if which == "w_bf16" else ttnn.bfloat8_b
    tx, tw = _to_device(x, w, device, x_dtype=x_dtype, w_dtype=w_dtype)
    cfg = None
    if which == "fp32_dest_acc":
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True)
    with expect_error(NotImplementedError, ""):
        wq_b_linear(tx, tw, compute_kernel_config=cfg)
