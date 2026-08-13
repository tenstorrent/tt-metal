# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm — IMMUTABLE acceptance test.

This file is the specification. The implementer MUST NOT modify it.

Run from repo root:
    scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm.py

Design: ttnn/ttnn/operations/rms_norm/op_design.md
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.rms_norm import rms_norm

# Same thresholds as the golden suite — keyed by dtype only.
PCC = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}

TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
}


def rms_norm_reference(x: torch.Tensor, gamma: torch.Tensor = None, epsilon: float = 1e-6) -> torch.Tensor:
    """out = x / sqrt(mean(x^2, dim=-1, keepdim=True) + eps) * gamma"""
    xf = x.to(torch.float32)
    denom = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    out = xf * denom
    if gamma is not None:
        out = out * gamma.to(torch.float32).reshape(-1)
    return out


def _make_inputs(shape, dtype, layout, with_gamma, device, seed=42):
    torch.manual_seed(seed)
    torch_dtype = TORCH_DTYPE[dtype]

    torch_x = torch.randn(shape, dtype=torch.float32).to(torch_dtype)
    ttnn_x = ttnn.from_torch(
        torch_x,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    torch_gamma = None
    ttnn_gamma = None
    if with_gamma:
        w = shape[-1]
        torch_gamma = torch.randn((1, 1, 1, w), dtype=torch.float32).to(torch_dtype)
        ttnn_gamma = ttnn.from_torch(
            torch_gamma,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return torch_x, ttnn_x, torch_gamma, ttnn_gamma


def _check(ttnn_out, shape, layout, torch_x, torch_gamma, dtype, epsilon=1e-6):
    assert list(ttnn_out.shape) == list(shape), f"shape mismatch: {ttnn_out.shape} vs {shape}"
    assert ttnn_out.layout == layout, f"layout mismatch: {ttnn_out.layout} vs {layout}"
    assert ttnn_out.dtype == dtype, f"dtype mismatch: {ttnn_out.dtype} vs {dtype}"

    actual = ttnn.to_torch(ttnn_out).to(torch.float32)
    expected = rms_norm_reference(torch_x, torch_gamma, epsilon)
    assert_with_pcc(expected, actual, PCC[dtype])


# ---------------------------------------------------------------------------
# Core sweep: shapes x layouts x dtypes x gamma
# ---------------------------------------------------------------------------

SHAPES = [
    # single tile
    pytest.param((32, 32), id="single_tile_2d"),
    # multi-tile, square
    pytest.param((64, 64), id="2x2_tiles_2d"),
    # non-square, wide (multi-tile reduce)
    pytest.param((32, 128), id="1x4_tiles_wide"),
    # non-square, tall
    pytest.param((256, 32), id="8x1_tiles_tall"),
    # multi-batch 3D
    pytest.param((2, 64, 128), id="batch2_3d"),
    # multi-batch 4D
    pytest.param((2, 4, 128, 256), id="batch2x4_4d"),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("with_gamma", [True, False], ids=["gamma", "no_gamma"])
def test_rms_norm(device, shape, layout, dtype, with_gamma):
    torch_x, ttnn_x, torch_gamma, ttnn_gamma = _make_inputs(shape, dtype, layout, with_gamma, device)
    out = rms_norm(ttnn_x, gamma=ttnn_gamma)
    _check(out, shape, layout, torch_x, torch_gamma, dtype)


# ---------------------------------------------------------------------------
# Non-tile-aligned shapes — H and/or W not a multiple of 32, both layouts.
# The RMS denominator must reflect only the valid (non-padding) elements.
# ---------------------------------------------------------------------------

NON_ALIGNED_SHAPES = [
    pytest.param((32, 50), id="w_non_aligned"),
    pytest.param((32, 17), id="w_non_aligned_sub_tile"),
    pytest.param((17, 64), id="h_non_aligned"),
    pytest.param((47, 100), id="hw_non_aligned"),
    pytest.param((2, 4, 47, 47), id="hw_non_aligned_4d"),
    # small W where one tile of padding is a large fraction of the row:
    # folding padding into the reduction is a >15% error here.
    pytest.param((32, 40), id="narrow_w_40"),
    pytest.param((32, 72), id="narrow_w_72"),
]


@pytest.mark.parametrize("shape", NON_ALIGNED_SHAPES)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_rms_norm_non_tile_aligned(device, shape, layout, dtype):
    torch_x, ttnn_x, torch_gamma, ttnn_gamma = _make_inputs(shape, dtype, layout, True, device)
    out = rms_norm(ttnn_x, gamma=ttnn_gamma)
    _check(out, shape, layout, torch_x, torch_gamma, dtype)


# ---------------------------------------------------------------------------
# Regime-pinned tests.
#
# The op selects between two compute regimes (op_design.md -> Work Distribution):
#   num_hidden_slices == 1  (RowParallel, no cross-core combine)
#   num_hidden_slices  > 1  (BlockParallel + cross-core partial-sum combine)
# The predicate reads the device grid, so a regime that only triggers on some
# grids passes on one device and fails on another. These shapes pin both sides.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        # Many tile-rows, narrow hidden -> rows over-fill the grid -> RowParallel.
        pytest.param((4096, 64), id="row_parallel_tall"),
        pytest.param((2048, 128), id="row_parallel_tall_wide"),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_rms_norm_regime_row_parallel(device, shape, dtype):
    torch_x, ttnn_x, torch_gamma, ttnn_gamma = _make_inputs(shape, dtype, ttnn.TILE_LAYOUT, True, device)
    out = rms_norm(ttnn_x, gamma=ttnn_gamma)
    _check(out, shape, ttnn.TILE_LAYOUT, torch_x, torch_gamma, dtype)


@pytest.mark.parametrize(
    "shape",
    [
        # One or few tile-rows with a wide hidden dim: rows cannot fill the grid,
        # so the hidden (reduced) axis must be split across cores and the partial
        # sums combined. These force the cross-core combine path.
        pytest.param((1, 1, 32, 4096), id="combine_1row_w4096"),
        pytest.param((1, 1, 32, 8192), id="combine_1row_w8192"),
        pytest.param((1, 1, 64, 12288), id="combine_2rows_w12288"),
        # cross-core combine together with a W mask (non-tile-aligned wide row)
        pytest.param((1, 1, 32, 4095), id="combine_1row_w4095_masked"),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_rms_norm_regime_cross_core_combine(device, shape, dtype):
    torch_x, ttnn_x, torch_gamma, ttnn_gamma = _make_inputs(shape, dtype, ttnn.TILE_LAYOUT, True, device)
    out = rms_norm(ttnn_x, gamma=ttnn_gamma)
    _check(out, shape, ttnn.TILE_LAYOUT, torch_x, torch_gamma, dtype)


# ---------------------------------------------------------------------------
# epsilon
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("epsilon", [1e-6, 1e-5, 1e-2])
def test_rms_norm_epsilon(device, epsilon):
    shape = (64, 128)
    torch_x, ttnn_x, torch_gamma, ttnn_gamma = _make_inputs(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, True, device)
    out = rms_norm(ttnn_x, gamma=ttnn_gamma, epsilon=epsilon)
    _check(out, shape, ttnn.TILE_LAYOUT, torch_x, torch_gamma, ttnn.bfloat16, epsilon=epsilon)


def test_rms_norm_epsilon_dominates(device):
    """Tiny inputs: epsilon dominates the denominator, so it must be added
    before the rsqrt and must not be dropped."""
    shape = (32, 64)
    epsilon = 1e-2
    torch.manual_seed(42)
    torch_x = (torch.randn(shape, dtype=torch.float32) * 1e-4).to(torch.bfloat16)
    ttnn_x = ttnn.from_torch(
        torch_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = rms_norm(ttnn_x, epsilon=epsilon)
    _check(out, shape, ttnn.TILE_LAYOUT, torch_x, None, ttnn.bfloat16, epsilon=epsilon)


# ---------------------------------------------------------------------------
# compute_kernel_config
# ---------------------------------------------------------------------------


def test_rms_norm_default_compute_kernel_config_is_exported():
    """The Phase 0 default must live in exactly one exported factory, and it
    must be a factory (fresh descriptor per call), not a shared constant."""
    from ttnn.operations.rms_norm import default_compute_kernel_config

    a = default_compute_kernel_config()
    b = default_compute_kernel_config()
    assert a is not b, "default_compute_kernel_config must be a factory, not a shared constant"
    assert a.fp32_dest_acc_en is True


@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.LoFi])
def test_rms_norm_math_fidelity_is_not_gated(device, math_fidelity):
    """math_fidelity is not a gated axis: any value is accepted and honored."""
    shape = (64, 128)
    torch_x, ttnn_x, torch_gamma, ttnn_gamma = _make_inputs(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, True, device)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=math_fidelity, fp32_dest_acc_en=True)
    out = rms_norm(ttnn_x, gamma=ttnn_gamma, compute_kernel_config=cfg)
    _check(out, shape, ttnn.TILE_LAYOUT, torch_x, torch_gamma, ttnn.bfloat16)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_rms_norm_maxed_out_precision_corner(device, dtype):
    """The Phase 0 precision corner spelled out explicitly."""
    shape = (2, 64, 256)
    torch_x, ttnn_x, torch_gamma, ttnn_gamma = _make_inputs(shape, dtype, ttnn.TILE_LAYOUT, True, device)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
    )
    out = rms_norm(ttnn_x, gamma=ttnn_gamma, compute_kernel_config=cfg)
    _check(out, shape, ttnn.TILE_LAYOUT, torch_x, torch_gamma, dtype)


def test_rms_norm_float32_rejects_non_fp32_dest_acc(device, expect_error):
    """float32 input with fp32_dest_acc_en=False is refused natively."""
    _, ttnn_x, _, _ = _make_inputs((64, 64), ttnn.float32, ttnn.TILE_LAYOUT, False, device)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False)
    with expect_error((ValueError, RuntimeError), "(?i)fp32_dest_acc_en|float32"):
        rms_norm(ttnn_x, compute_kernel_config=cfg)


# ---------------------------------------------------------------------------
# gamma format independence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gamma_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["g_tile", "g_rm"])
@pytest.mark.parametrize("gamma_dtype", [ttnn.bfloat16, ttnn.float32], ids=["g_bf16", "g_fp32"])
def test_rms_norm_gamma_formats(device, gamma_layout, gamma_dtype):
    """gamma may be at a different dtype/layout than the input (bf16
    activations + fp32 weights is a common mixed-precision LLM case)."""
    shape = (64, 128)
    torch.manual_seed(42)
    torch_x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    ttnn_x = ttnn.from_torch(
        torch_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    torch_gamma = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32).to(TORCH_DTYPE[gamma_dtype])
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=gamma_dtype,
        layout=gamma_layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = rms_norm(ttnn_x, gamma=ttnn_gamma)
    _check(out, shape, ttnn.TILE_LAYOUT, torch_x, torch_gamma, ttnn.bfloat16)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rms_norm_rejects_rank_1(device, expect_error):
    torch.manual_seed(42)
    t = ttnn.from_torch(
        torch.randn((64,), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with expect_error((ValueError, RuntimeError), "(?i)rank|dimension|at least 2"):
        rms_norm(t)


def test_rms_norm_rejects_gamma_width_mismatch(device, expect_error):
    torch.manual_seed(42)
    x = ttnn.from_torch(
        torch.randn((64, 128), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    bad_gamma = ttnn.from_torch(
        torch.randn((1, 1, 1, 64), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with expect_error((ValueError, RuntimeError), "(?i)gamma|last dim|width"):
        rms_norm(x, gamma=bad_gamma)


# ---------------------------------------------------------------------------
# The entry point performs no host-side layout/shape workaround.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
def test_rms_norm_no_host_side_transform(device, layout, monkeypatch):
    """The Python entry point must not call to_layout / tilize / untilize /
    pad / slice to work around layout or alignment differences."""
    shape = (47, 100)  # non-tile-aligned in both H and W
    # Build the inputs BEFORE installing the guards: from_torch itself is allowed
    # to lay out a tensor, only the op's entry point is under test.
    torch_x, ttnn_x, torch_gamma, ttnn_gamma = _make_inputs(shape, ttnn.bfloat16, layout, True, device)

    banned = ["to_layout", "tilize", "untilize", "pad", "slice"]
    calls = []

    def make_guard(n, orig):
        def guard(*args, **kwargs):
            calls.append(n)
            return orig(*args, **kwargs)

        return guard

    for name in banned:
        original = getattr(ttnn, name, None)
        if original is None:
            continue
        monkeypatch.setattr(ttnn, name, make_guard(name, original))

    out = rms_norm(ttnn_x, gamma=ttnn_gamma)

    assert not calls, f"entry point used host-side transform(s): {sorted(set(calls))}"
    _check(out, shape, layout, torch_x, torch_gamma, ttnn.bfloat16)
