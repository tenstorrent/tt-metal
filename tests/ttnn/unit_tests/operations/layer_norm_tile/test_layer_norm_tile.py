# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for layer_norm_tile (immutable spec; do not modify).

Input: bf16, TILE_LAYOUT, L1 interleaved, shape (B, 1, S, 768) with 1024 <= B*S <= 4096, S <= 512,
S not necessarily tile-aligned. Gamma/beta are required, in one of two production layouts:
(768,) TILE in DRAM, or (1, 1, 24, 32) ROW_MAJOR in L1.
"""

import pytest
import torch

import ttnn
from ttnn.operations.layer_norm_tile import layer_norm

HIDDEN = 768

PCC_THRESHOLD = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}

GAMMA_VARIANTS = {
    "tile_dram": ((HIDDEN,), ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
    "rm_l1": ((1, 1, HIDDEN // 32, 32), ttnn.ROW_MAJOR_LAYOUT, ttnn.L1_MEMORY_CONFIG),
    "tile_dram_4d": ((1, 1, 1, HIDDEN), ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
    "rm_l1_flat": ((1, 1, 1, HIDDEN), ttnn.ROW_MAJOR_LAYOUT, ttnn.L1_MEMORY_CONFIG),
}

TILE_ALIGNED_SHAPES = [
    (32, 1, 32, HIDDEN),  # one tile-row per image, 32 tile-rows
    (8, 1, 128, HIDDEN),
    (4, 1, 256, HIDDEN),
    (2, 1, 512, HIDDEN),
    (16, 1, 256, HIDDEN),  # 128 tile-rows: some cores own 2 tile-rows
    (8, 1, 512, HIDDEN),
]

NON_TILE_ALIGNED_SHAPES = [
    (8, 1, 200, HIDDEN),
    (3, 1, 500, HIDDEN),
    (20, 1, 100, HIDDEN),
    (64, 1, 17, HIDDEN),
    (7, 1, 300, HIDDEN),
    (3, 1, 342, HIDDEN),
]

ALL_SHAPES = TILE_ALIGNED_SHAPES + NON_TILE_ALIGNED_SHAPES


def _shape_id(shape):
    return "x".join(str(d) for d in shape)


def _pcc(expected, actual):
    expected = expected.flatten().double()
    actual = actual.flatten().double()
    stacked = torch.stack([expected, actual])
    return torch.corrcoef(stacked)[0, 1].item()


def _reference(x, gamma, beta, residual, epsilon):
    xf = x.float()
    if residual is not None:
        xf = xf + residual.float()
    return torch.nn.functional.layer_norm(
        xf, (HIDDEN,), weight=gamma.float().reshape(HIDDEN), bias=beta.float().reshape(HIDDEN), eps=epsilon
    )


def _run_case(
    device,
    shape,
    gamma_variant,
    with_residual,
    *,
    epsilon=None,
    compute_kernel_config=None,
    residual_positional=False,
):
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.bfloat16)
    residual = torch.randn(shape, dtype=torch.bfloat16) if with_residual else None
    gamma_shape, gamma_layout, gamma_mem = GAMMA_VARIANTS[gamma_variant]
    gamma = torch.randn(gamma_shape, dtype=torch.bfloat16)
    beta = torch.randn(gamma_shape, dtype=torch.bfloat16)

    tt_x = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tt_gamma = ttnn.from_torch(gamma, dtype=ttnn.bfloat16, layout=gamma_layout, device=device, memory_config=gamma_mem)
    tt_beta = ttnn.from_torch(beta, dtype=ttnn.bfloat16, layout=gamma_layout, device=device, memory_config=gamma_mem)
    tt_residual = None
    if with_residual:
        tt_residual = ttnn.from_torch(
            residual, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )

    kwargs = {}
    if epsilon is not None:
        kwargs["epsilon"] = epsilon
    if compute_kernel_config is not None:
        kwargs["compute_kernel_config"] = compute_kernel_config

    if tt_residual is None:
        tt_out = layer_norm(tt_x, tt_gamma, tt_beta, **kwargs)
    elif residual_positional:
        tt_out = layer_norm(tt_x, tt_gamma, tt_beta, tt_residual, **kwargs)
    else:
        tt_out = layer_norm(tt_x, tt_gamma, tt_beta, residual_input_tensor=tt_residual, **kwargs)

    assert tt_out.layout == ttnn.TILE_LAYOUT
    assert tt_out.dtype == ttnn.bfloat16
    assert tuple(tt_out.shape) == tuple(shape)
    out_mem = tt_out.memory_config()
    assert out_mem.buffer_type == ttnn.BufferType.L1
    assert out_mem.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED

    actual = ttnn.to_torch(tt_out).float()
    expected = _reference(x, gamma, beta, residual, 1e-5 if epsilon is None else epsilon)

    for t in (tt_out, tt_x, tt_gamma, tt_beta, tt_residual):
        if t is not None:
            ttnn.deallocate(t)

    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all(), "output contains non-finite values"
    pcc = _pcc(expected, actual)
    threshold = PCC_THRESHOLD[ttnn.bfloat16]
    assert (
        pcc >= threshold
    ), f"PCC {pcc:.6f} < {threshold} for shape={shape} gamma={gamma_variant} residual={with_residual}"


@pytest.mark.parametrize("gamma_variant", ["tile_dram", "rm_l1"])
@pytest.mark.parametrize("with_residual", [False, True], ids=["no_residual", "residual"])
@pytest.mark.parametrize("shape", ALL_SHAPES, ids=_shape_id)
def test_layer_norm_tile(device, shape, with_residual, gamma_variant):
    _run_case(device, shape, gamma_variant, with_residual)


@pytest.mark.parametrize("gamma_variant", ["tile_dram_4d", "rm_l1_flat"])
@pytest.mark.parametrize("with_residual", [False, True], ids=["no_residual", "residual"])
@pytest.mark.parametrize("shape", [(8, 1, 128, HIDDEN), (3, 1, 500, HIDDEN)], ids=_shape_id)
def test_layer_norm_tile_affine_shapes(device, shape, with_residual, gamma_variant):
    _run_case(device, shape, gamma_variant, with_residual)


COMPUTE_CONFIGS = {
    "hifi4_fp32": lambda: ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
    "hifi4_bf16_dest": lambda: ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False
    ),
    "lofi": lambda: ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi),
    "hifi2_fp32_approx": lambda: ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, math_approx_mode=True
    ),
}


@pytest.mark.parametrize("config_name", list(COMPUTE_CONFIGS))
@pytest.mark.parametrize("with_residual", [False, True], ids=["no_residual", "residual"])
@pytest.mark.parametrize("shape", [(8, 1, 128, HIDDEN), (5, 1, 333, HIDDEN)], ids=_shape_id)
def test_layer_norm_tile_compute_config(device, shape, with_residual, config_name):
    _run_case(device, shape, "rm_l1", with_residual, compute_kernel_config=COMPUTE_CONFIGS[config_name]())


@pytest.mark.parametrize("epsilon", [1e-6, 1e-3])
def test_layer_norm_tile_epsilon_full_config(device, epsilon):
    _run_case(
        device,
        (8, 1, 128, HIDDEN),
        "tile_dram",
        True,
        epsilon=epsilon,
        compute_kernel_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
        residual_positional=True,
    )


# Regime-pinned: 256 tile-rows exceed the single-block capacity per core on any grid with fewer
# than 128 cores (fp32 DEST + residual), so this exercises the multi-block, double-buffered path.
@pytest.mark.parametrize("gamma_variant", ["tile_dram", "rm_l1"])
@pytest.mark.parametrize("with_residual", [False, True], ids=["no_residual", "residual"])
def test_layer_norm_tile_multi_block(device, with_residual, gamma_variant):
    _run_case(device, (256, 1, 16, HIDDEN), gamma_variant, with_residual)


def _affine(device):
    gamma = ttnn.from_torch(
        torch.ones((HIDDEN,), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta = ttnn.from_torch(
        torch.zeros((HIDDEN,), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return gamma, beta


def test_layer_norm_tile_rejects_float32(device, expect_error):
    torch.manual_seed(42)
    x = ttnn.from_torch(
        torch.randn((8, 1, 128, HIDDEN), dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    gamma, beta = _affine(device)
    with expect_error(ValueError, ""):
        layer_norm(x, gamma, beta)


def test_layer_norm_tile_rejects_row_major_input(device, expect_error):
    torch.manual_seed(42)
    x = ttnn.from_torch(
        torch.randn((8, 1, 128, HIDDEN), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    gamma, beta = _affine(device)
    with expect_error((ValueError, NotImplementedError), ""):
        layer_norm(x, gamma, beta)
