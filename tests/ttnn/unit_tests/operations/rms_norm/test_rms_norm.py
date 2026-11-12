# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Acceptance test for rms_norm (immutable spec — do not modify).

RMSNorm(x) = x / sqrt(mean(x^2, dim=-1, keepdim=True) + epsilon) * gamma

Phase 0 corner: fp32_dest_acc_en=True, interleaved DRAM, bf16/fp32, TILE and
ROW_MAJOR, tile-aligned and non-aligned H/W, with and without gamma.
Accuracy gate: PCC + relative RMS from the golden suite's helpers.TOLERANCES.
"""

import pytest
import torch
import ttnn

from eval.metrics import check_output
from eval.golden_tests.rms_norm.helpers import TOLERANCES

from ttnn.operations.rms_norm import rms_norm


_TORCH_DTYPE = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}


def torch_rms_norm(x, gamma=None, epsilon=1e-6):
    """Reference in fp32, returned in the input dtype."""
    xf = x.to(torch.float32)
    out = xf / torch.sqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + epsilon)
    if gamma is not None:
        out = out * gamma.to(torch.float32).reshape(-1)
    return out.to(x.dtype)


def _config(fp32_dest_acc_en=True):
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = fp32_dest_acc_en
    cfg.math_approx_mode = False
    return cfg


def _run(device, shape, dtype, layout, with_gamma, epsilon=1e-6, compute_kernel_config="explicit"):
    torch.manual_seed(42)
    torch_dtype = _TORCH_DTYPE[dtype]
    W = shape[-1]

    torch_input = torch.randn(shape, dtype=torch.float32).to(torch_dtype)
    torch_gamma = torch.randn(W, dtype=torch.float32).to(torch_dtype) if with_gamma else None

    tt_input = ttnn.from_torch(
        torch_input, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_gamma = None
    if with_gamma:
        tt_gamma = ttnn.from_torch(
            torch_gamma.reshape(1, 1, 1, W),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    kwargs = {"gamma": tt_gamma, "epsilon": epsilon}
    if compute_kernel_config == "explicit":
        kwargs["compute_kernel_config"] = _config(True)
    elif compute_kernel_config is not None:
        kwargs["compute_kernel_config"] = compute_kernel_config

    tt_output = rms_norm(tt_input, **kwargs)

    expected = torch_rms_norm(torch_input, torch_gamma, epsilon)
    check_output(
        tt_output,
        expected,
        shape=list(shape),
        dtype=dtype,
        expected_layout=layout,
        tolerance=TOLERANCES[dtype],
    )


SHAPES = [
    (1, 1, 32, 32),  # single tile
    (1, 1, 64, 128),  # multi-tile
    (1, 1, 32, 50),  # W non-aligned
    (1, 1, 47, 64),  # H non-aligned
    (1, 1, 17, 50),  # both non-aligned
    (2, 4, 64, 256),  # multi-batch
    (4, 8, 32, 47),  # multi-batch, W non-aligned (> grid tile-rows)
    (1, 1, 2048, 256),  # many tile-rows -> several per core
    (32, 64),  # rank 2
    (1, 17, 100),  # rank 3, both non-aligned
    (1, 1, 32, 4096),  # wide row (one tile-row)
    (1, 32, 8192),  # widest INPUTS row (W chunking under L1 pressure)
]


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("with_gamma", [True, False], ids=["gamma", "no_gamma"])
def test_rms_norm(device, shape, layout, dtype, with_gamma):
    _run(device, shape, dtype, layout, with_gamma)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize("epsilon", [1e-5, 1e-2])
def test_rms_norm_epsilon(device, layout, epsilon):
    _run(device, (1, 1, 64, 96), ttnn.bfloat16, layout, True, epsilon=epsilon)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
def test_rms_norm_default_config(device, layout):
    """compute_kernel_config=None resolves through default_compute_kernel_config()."""
    _run(device, (2, 1, 64, 128), ttnn.float32, layout, True, compute_kernel_config=None)


def test_default_compute_kernel_config_is_phase0_corner():
    from ttnn.operations.rms_norm import default_compute_kernel_config

    cfg = default_compute_kernel_config()
    assert cfg.fp32_dest_acc_en is True
    assert cfg.math_fidelity == ttnn.MathFidelity.HiFi4


def test_rms_norm_rejects_rank1(device, expect_error):
    x = ttnn.from_torch(torch.randn(64), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error((ValueError, RuntimeError), ""):
        rms_norm(x)


def test_rms_norm_rejects_gamma_width_mismatch(device, expect_error):
    x = ttnn.from_torch(torch.randn(1, 1, 32, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    g = ttnn.from_torch(torch.randn(1, 1, 1, 32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error((ValueError, RuntimeError), ""):
        rms_norm(x, gamma=g)


def test_rms_norm_rejects_fp32_without_fp32_dest(device, expect_error):
    x = ttnn.from_torch(torch.randn(1, 1, 32, 64), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error((NotImplementedError, ValueError, RuntimeError), ""):
        rms_norm(x, compute_kernel_config=_config(fp32_dest_acc_en=False))
