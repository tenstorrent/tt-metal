# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for ttnn.operations.rms_norm (immutable spec — do not edit to make it pass).

RMSNorm(x) = x / sqrt(mean(x^2, dim=-1, keepdim=True) + epsilon) * gamma

Covers (op_design.md):
- both layouts (TILE / ROW_MAJOR) natively, output layout == input layout;
- tile-aligned, W-non-aligned, H-non-aligned and both-non-aligned shapes, ranks 2 / 3 / 4;
- bfloat16 and float32 at the Phase 0 precision corner (fp32_dest_acc_en=True);
- gamma absent / ROW_MAJOR (1,1,1,W) / TILE (1,1,1,W) / dtype different from the input;
- the mandated call patterns and the explicit compute_kernel_config corner;
- regime-pinned shapes: R1 row-local (single chunk and chunked rows), R2 W-split combine (grid-fill and L1-forced);
- the required validation errors.
"""

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm

PCC_THRESHOLD = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}

TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
}


# ----------------------------------------------------------------------------- helpers


def torch_rms_norm(x, gamma=None, epsilon=1e-6):
    """Reference in fp32; gamma broadcast over the last dimension."""
    xf = x.to(torch.float32)
    out = xf / torch.sqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + epsilon)
    if gamma is not None:
        out = out * gamma.to(torch.float32).reshape(-1)
    return out


def pcc(expected, actual):
    e = expected.to(torch.float32).flatten()
    a = actual.to(torch.float32).flatten()
    assert torch.isfinite(a).all(), "output contains non-finite values"
    if torch.equal(e, a):
        return 1.0
    e = e - e.mean()
    a = a - a.mean()
    denom = torch.sqrt((e * e).sum()) * torch.sqrt((a * a).sum())
    if denom == 0:
        return 0.0
    return float((e * a).sum() / denom)


def to_device(t, device, dtype, layout):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def make_gamma(W, device, dtype, layout):
    torch_gamma = torch.randn((1, 1, 1, W), dtype=torch.float32).to(TORCH_DTYPE[dtype])
    return torch_gamma, to_device(torch_gamma, device, dtype, layout)


def run_and_check(
    device,
    shape,
    dtype,
    layout,
    *,
    gamma_layout=None,
    gamma_dtype=None,
    epsilon=None,
    compute_kernel_config=None,
):
    torch.manual_seed(42)
    torch_x = torch.randn(shape, dtype=torch.float32).to(TORCH_DTYPE[dtype])
    tt_x = to_device(torch_x, device, dtype, layout)

    torch_gamma, tt_gamma = None, None
    if gamma_layout is not None:
        torch_gamma, tt_gamma = make_gamma(shape[-1], device, gamma_dtype or dtype, gamma_layout)

    kwargs = {}
    if tt_gamma is not None:
        kwargs["gamma"] = tt_gamma
    if epsilon is not None:
        kwargs["epsilon"] = epsilon
    if compute_kernel_config is not None:
        kwargs["compute_kernel_config"] = compute_kernel_config

    tt_out = rms_norm(tt_x, **kwargs)

    assert tt_out.layout == layout, f"output layout {tt_out.layout} != input layout {layout}"
    assert tt_out.dtype == dtype, f"output dtype {tt_out.dtype} != input dtype {dtype}"
    assert list(tt_out.shape) == list(shape), f"output shape {list(tt_out.shape)} != {list(shape)}"

    expected = torch_rms_norm(torch_x, torch_gamma, 1e-6 if epsilon is None else epsilon)
    actual = ttnn.to_torch(tt_out)
    assert list(actual.shape) == list(shape)
    value = pcc(expected, actual)
    assert value >= PCC_THRESHOLD[dtype], f"PCC {value:.6f} < {PCC_THRESHOLD[dtype]} for {shape} {dtype} {layout}"


# ----------------------------------------------------------------------------- main matrix

SHAPES = [
    pytest.param((1, 1, 32, 32), id="single_tile"),
    pytest.param((1, 1, 64, 128), id="multi_tile"),
    pytest.param((1, 1, 32, 512), id="non_square_wide"),
    pytest.param((2, 4, 64, 96), id="multi_batch"),
    pytest.param((1, 1, 47, 50), id="both_non_aligned"),
    pytest.param((4, 128, 47), id="rank3_w_non_aligned"),
    pytest.param((17, 64), id="rank2_h_non_aligned"),
    pytest.param((128, 512), id="rank2_aligned"),
]

LAYOUTS = [
    pytest.param(ttnn.TILE_LAYOUT, id="TILE"),
    pytest.param(ttnn.ROW_MAJOR_LAYOUT, id="RM"),
]

DTYPES = [
    pytest.param(ttnn.bfloat16, id="bf16"),
    pytest.param(ttnn.float32, id="fp32"),
]


@pytest.mark.parametrize("with_gamma", [False, True], ids=["no_gamma", "rm_gamma"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("shape", SHAPES)
def test_rms_norm(device, shape, layout, dtype, with_gamma):
    run_and_check(
        device,
        shape,
        dtype,
        layout,
        gamma_layout=ttnn.ROW_MAJOR_LAYOUT if with_gamma else None,
    )


# ----------------------------------------------------------------------------- gamma formats


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 64, 128), id="aligned"),
        pytest.param((1, 1, 32, 72), id="w_non_aligned"),
        pytest.param((2, 1, 100, 47), id="both_non_aligned"),
    ],
)
def test_rms_norm_tile_gamma(device, shape, layout):
    run_and_check(device, shape, ttnn.bfloat16, layout, gamma_layout=ttnn.TILE_LAYOUT)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    "dtype, gamma_dtype",
    [
        pytest.param(ttnn.bfloat16, ttnn.float32, id="bf16_in_fp32_gamma"),
        pytest.param(ttnn.float32, ttnn.bfloat16, id="fp32_in_bf16_gamma"),
    ],
)
def test_rms_norm_mixed_gamma_dtype(device, layout, dtype, gamma_dtype):
    run_and_check(
        device,
        (1, 1, 64, 160),
        dtype,
        layout,
        gamma_layout=ttnn.ROW_MAJOR_LAYOUT,
        gamma_dtype=gamma_dtype,
    )


# ----------------------------------------------------------------------------- call patterns / config


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    "with_gamma, epsilon",
    [
        pytest.param(False, None, id="rms_norm(x)"),
        pytest.param(True, None, id="rms_norm(x,gamma)"),
        pytest.param(False, 1e-5, id="rms_norm(x,epsilon)"),
        pytest.param(True, 1e-5, id="rms_norm(x,gamma,epsilon)"),
        pytest.param(True, 1e-2, id="large_epsilon"),
    ],
)
def test_rms_norm_call_patterns(device, layout, with_gamma, epsilon):
    run_and_check(
        device,
        (1, 1, 64, 256),
        ttnn.bfloat16,
        layout,
        gamma_layout=ttnn.ROW_MAJOR_LAYOUT if with_gamma else None,
        epsilon=epsilon,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_rms_norm_explicit_compute_kernel_config(device, layout, dtype):
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
    )
    run_and_check(
        device,
        (2, 1, 64, 320),
        dtype,
        layout,
        gamma_layout=ttnn.ROW_MAJOR_LAYOUT,
        compute_kernel_config=config,
    )


# ----------------------------------------------------------------------------- regime-pinned shapes
# The shapes below select the same regime on any grid of <= 256 Tensix cores with <= 1.5 MiB L1 per core
# (op_design.md, Work Distribution: regime-pinned tests).


@pytest.mark.parametrize("layout", LAYOUTS)
def test_rms_norm_regime_row_local_many_blocks(device, layout):
    # R1: 256 tile-rows >= cores, tiny W slice -> row split only, several blocks per core.
    run_and_check(device, (1, 1, 8192, 64), ttnn.bfloat16, layout, gamma_layout=ttnn.ROW_MAJOR_LAYOUT)


def test_rms_norm_regime_row_local_chunked_rows(device):
    # R1 with chunks_per_row > 1: the whole-slice streaming windows do not fit next to the resident row.
    run_and_check(device, (1, 1, 8192, 4096), ttnn.bfloat16, ttnn.TILE_LAYOUT, gamma_layout=ttnn.ROW_MAJOR_LAYOUT)


@pytest.mark.parametrize("with_gamma", [False, True], ids=["no_gamma", "rm_gamma"])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 16384), id="W16384_aligned"),
        pytest.param((1, 1, 32, 16300), id="W16300_non_aligned"),
        pytest.param((1, 1, 64, 12288), id="two_tile_rows_W12288"),
    ],
)
def test_rms_norm_regime_w_split_fill(device, shape, layout, with_gamma):
    # R2: one or two tile-rows, wide W -> W split across a group of cores with the root combine.
    run_and_check(
        device,
        shape,
        ttnn.bfloat16,
        layout,
        gamma_layout=ttnn.ROW_MAJOR_LAYOUT if with_gamma else None,
    )


@pytest.mark.parametrize("layout", LAYOUTS)
def test_rms_norm_decode_shape(device, layout):
    # LLM decode: one tile-row of a 7168-wide hidden dimension (R2).
    run_and_check(device, (1, 1, 32, 7168), ttnn.bfloat16, layout, gamma_layout=ttnn.ROW_MAJOR_LAYOUT)


def test_rms_norm_regime_w_split_l1_forced(device):
    # R2 forced by L1: 128 tile-rows fill the grid, but a full fp32 8192-wide row plus fp32 gamma does not fit,
    # so W is split and each core runs several blocks, each with its own combine round.
    run_and_check(
        device,
        (1, 1, 4096, 8192),
        ttnn.float32,
        ttnn.TILE_LAYOUT,
        gamma_layout=ttnn.ROW_MAJOR_LAYOUT,
        gamma_dtype=ttnn.float32,
    )


# ----------------------------------------------------------------------------- validation


# The spec fixes the exception types, not the message text, so any message is accepted.
ANY_MESSAGE = r"(?s).*"


def test_rms_norm_rejects_rank_1(device, expect_error):
    tt_x = to_device(
        torch.randn((64,), dtype=torch.float32).to(torch.bfloat16), device, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT
    )
    with expect_error((ValueError, RuntimeError), ANY_MESSAGE):
        rms_norm(tt_x)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_rms_norm_rejects_gamma_width_mismatch(device, layout, expect_error):
    torch.manual_seed(42)
    tt_x = to_device(torch.randn((1, 1, 32, 64), dtype=torch.float32).to(torch.bfloat16), device, ttnn.bfloat16, layout)
    _, tt_gamma = make_gamma(96, device, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    with expect_error((ValueError, RuntimeError), ANY_MESSAGE):
        rms_norm(tt_x, gamma=tt_gamma)
