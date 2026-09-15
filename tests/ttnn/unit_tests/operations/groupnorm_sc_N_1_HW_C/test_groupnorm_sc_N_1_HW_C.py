# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for groupnorm_sc_N_1_HW_C — the immutable specification.

Do NOT modify this file when implementing the op. It encodes:

- the exact import path and signature (torch.nn.GroupNorm-shaped, no mask args),
- output dtype == input dtype, output layout == TILE_LAYOUT (for RM input too),
- the partial-channel regime ((C / num_groups) % 32 != 0) as a first-class case,
- both input layouts, all affine call patterns, custom eps,
- multi-core regime pins (forced streaming; single core per image),
- ValueError argument validation.

Tolerances are the golden-suite PCC thresholds keyed by dtype.
"""

import pytest
import torch
import ttnn

from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C

PCC_BY_DTYPE = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}

TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.bfloat16,
}


def pytorch_reference(x, num_groups, *, gamma=None, beta=None, eps=1e-5):
    """GroupNorm on (N, 1, HW, C): permute to (N, C, HW), run torch, permute back. fp32 math."""
    original_dtype = x.dtype
    xf = x.to(torch.float32)
    N, one, HW, C = xf.shape
    assert one == 1
    x_nchw = xf.squeeze(1).permute(0, 2, 1)
    weight = gamma.to(torch.float32).reshape(C) if gamma is not None else None
    bias = beta.to(torch.float32).reshape(C) if beta is not None else None
    out = torch.nn.functional.group_norm(x_nchw, num_groups, weight=weight, bias=bias, eps=eps)
    return out.permute(0, 2, 1).unsqueeze(1).to(original_dtype)


def pcc(a, b):
    a = a.to(torch.float64).flatten()
    b = b.to(torch.float64).flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0 if torch.equal(a, b) else 0.0
    return float((a @ b) / denom)


def _make_input(shape, dtype, layout, device):
    torch_x = torch.randn(shape, dtype=torch.float32).to(TORCH_DTYPE[dtype])
    ttnn_x = ttnn.from_torch(torch_x, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return torch_x, ttnn_x


def _make_affine(C, affine_dtype, affine_layout, device):
    t = torch.randn((1, 1, 1, C), dtype=torch.float32).to(TORCH_DTYPE[affine_dtype])
    tt = ttnn.from_torch(
        t, dtype=affine_dtype, layout=affine_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return t, tt


def _check(output, expected, shape, dtype, layout_expected=ttnn.TILE_LAYOUT):
    assert output.dtype == dtype, f"output dtype {output.dtype} != input dtype {dtype}"
    assert output.layout == layout_expected, f"output layout {output.layout} != {layout_expected}"
    assert list(output.shape) == list(shape), f"shape mismatch: {output.shape} vs {shape}"
    got = ttnn.to_torch(output).to(torch.float32)
    exp = expected.to(torch.float32)
    assert torch.isfinite(got).all(), "output contains NaN/Inf"
    p = pcc(got, exp)
    assert p >= PCC_BY_DTYPE[dtype], f"PCC {p:.6f} < {PCC_BY_DTYPE[dtype]} (dtype={dtype})"


# (shape, num_groups) — tile-aligned group-aligned, group-straddling (SD/SDXL), multi-batch, non-square.
SHAPES = [
    pytest.param((1, 1, 32, 32), 1, id="single_tile_g1"),
    pytest.param((1, 1, 64, 128), 4, id="multi_tile_g4_aligned"),
    pytest.param((1, 1, 128, 64), 2, id="non_square_tall_g2"),
    pytest.param((1, 1, 32, 512), 16, id="non_square_wide_g16"),
    pytest.param((2, 1, 64, 128), 4, id="batch2_g4"),
    pytest.param((4, 1, 128, 256), 8, id="batch4_g8"),
    pytest.param((1, 1, 64, 320), 32, id="straddle_cg10"),
    pytest.param((1, 1, 64, 160), 8, id="straddle_cg20"),
    pytest.param((1, 1, 256, 1280), 32, id="straddle_cg40_sd"),
    pytest.param((1, 1, 64, 384), 8, id="straddle_cg48"),
    pytest.param((2, 1, 128, 192), 8, id="batch2_straddle_cg24"),
    pytest.param((1, 1, 1024, 640), 32, id="sd15_hw1024_c640"),
]


@pytest.mark.parametrize("shape,num_groups", SHAPES)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
def test_groupnorm_no_affine(device, shape, num_groups, layout, dtype):
    torch.manual_seed(42)
    torch_x, ttnn_x = _make_input(shape, dtype, layout, device)
    expected = pytorch_reference(torch_x, num_groups)
    output = groupnorm_sc_N_1_HW_C(ttnn_x, num_groups)
    _check(output, expected, shape, dtype)


@pytest.mark.parametrize(
    "shape,num_groups",
    [
        pytest.param((1, 1, 64, 128), 4, id="aligned_g4"),
        pytest.param((2, 1, 64, 320), 32, id="batch2_straddle_cg10"),
        pytest.param((1, 1, 256, 1280), 32, id="straddle_cg40_sd"),
        pytest.param((1, 1, 128, 448), 8, id="straddle_cg56"),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize("affine_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["affine_rm", "affine_tile"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
def test_groupnorm_gamma_beta(device, shape, num_groups, layout, affine_layout, dtype):
    torch.manual_seed(42)
    C = shape[-1]
    torch_x, ttnn_x = _make_input(shape, dtype, layout, device)
    torch_g, ttnn_g = _make_affine(C, dtype, affine_layout, device)
    torch_b, ttnn_b = _make_affine(C, dtype, affine_layout, device)
    expected = pytorch_reference(torch_x, num_groups, gamma=torch_g, beta=torch_b)
    output = groupnorm_sc_N_1_HW_C(ttnn_x, num_groups, gamma=ttnn_g, beta=ttnn_b)
    _check(output, expected, shape, dtype)


@pytest.mark.parametrize(
    "shape,num_groups",
    [
        pytest.param((1, 1, 64, 64), 2, id="aligned_g2"),
        pytest.param((1, 1, 64, 320), 32, id="straddle_cg10"),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
def test_groupnorm_gamma_only(device, shape, num_groups, dtype):
    torch.manual_seed(42)
    C = shape[-1]
    torch_x, ttnn_x = _make_input(shape, dtype, ttnn.TILE_LAYOUT, device)
    torch_g, ttnn_g = _make_affine(C, dtype, ttnn.ROW_MAJOR_LAYOUT, device)
    expected = pytorch_reference(torch_x, num_groups, gamma=torch_g)
    output = groupnorm_sc_N_1_HW_C(ttnn_x, num_groups, gamma=ttnn_g)
    _check(output, expected, shape, dtype)


@pytest.mark.parametrize("eps", [1e-6, 1e-3], ids=["eps1e-6", "eps1e-3"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
def test_groupnorm_custom_eps(device, eps, dtype):
    torch.manual_seed(42)
    shape, num_groups = (1, 1, 128, 640), 32  # C/G = 20 (straddling)
    C = shape[-1]
    torch_x, ttnn_x = _make_input(shape, dtype, ttnn.TILE_LAYOUT, device)
    # Small-variance data so eps is not negligible against var.
    torch_x = (torch_x.to(torch.float32) * 0.01).to(TORCH_DTYPE[dtype])
    ttnn_x = ttnn.from_torch(
        torch_x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    torch_g, ttnn_g = _make_affine(C, dtype, ttnn.ROW_MAJOR_LAYOUT, device)
    torch_b, ttnn_b = _make_affine(C, dtype, ttnn.ROW_MAJOR_LAYOUT, device)
    expected = pytorch_reference(torch_x, num_groups, gamma=torch_g, beta=torch_b, eps=eps)
    output = groupnorm_sc_N_1_HW_C(ttnn_x, num_groups, gamma=ttnn_g, beta=ttnn_b, eps=eps)
    _check(output, expected, shape, dtype)


def test_groupnorm_deterministic(device):
    """Same input twice → bit-identical output (no uninitialised L1 leaking into the statistics)."""
    torch.manual_seed(42)
    shape, num_groups = (1, 1, 64, 320), 32
    torch_x, ttnn_x = _make_input(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    out1 = ttnn.to_torch(groupnorm_sc_N_1_HW_C(ttnn_x, num_groups))
    out2 = ttnn.to_torch(groupnorm_sc_N_1_HW_C(ttnn_x, num_groups))
    assert torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# Regime pins. The design selects among regimes on the host (resident_2d /
# streaming_2d / single_core_per_image). The public signature has no knob
# arguments, so the pins go through the module-level knob overrides the
# implementer exposes for exactly this purpose (see op_design.md → Regimes:
# "regime-pinned tests are required"). The names below are the contract.
# ---------------------------------------------------------------------------


def _knob(name):
    import ttnn.operations.groupnorm_sc_N_1_HW_C as mod

    if not hasattr(mod, name):
        pytest.fail(f"op module must expose the `{name}` knob override for regime-pinned tests")
    return mod


def test_regime_streaming_forced(device):
    """Force the streaming regime (input read twice) on a shape that is resident by default."""
    mod = _knob("set_l1_budget_bytes_override")
    torch.manual_seed(42)
    shape, num_groups = (1, 1, 1024, 640), 32
    torch_x, ttnn_x = _make_input(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    expected = pytorch_reference(torch_x, num_groups)
    mod.set_l1_budget_bytes_override(0)  # nothing is resident → streaming_2d everywhere
    try:
        output = groupnorm_sc_N_1_HW_C(ttnn_x, num_groups)
    finally:
        mod.set_l1_budget_bytes_override(None)
    _check(output, expected, shape, ttnn.bfloat16)


def test_regime_single_core_per_image(device):
    """N >= num_cores → every core owns whole images, the combine degenerates to a local copy."""
    mod = _knob("set_max_cores_override")
    torch.manual_seed(42)
    shape, num_groups = (8, 1, 64, 160), 8  # C/G = 20 (straddling), 8 images
    torch_x, ttnn_x = _make_input(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    expected = pytorch_reference(torch_x, num_groups)
    mod.set_max_cores_override(4)  # 8 images over 4 cores → 2 images per core, P_n = 1
    try:
        output = groupnorm_sc_N_1_HW_C(ttnn_x, num_groups)
    finally:
        mod.set_max_cores_override(None)
    _check(output, expected, shape, ttnn.bfloat16)


# ---------------------------------------------------------------------------
# Argument validation (ValueError, not NotImplementedError). The message
# substrings below are part of the contract (op_design.md → Registry model):
# "rank", "dim", "num_groups", "gamma".
# ---------------------------------------------------------------------------


def test_rejects_wrong_rank(device, expect_error):
    x = ttnn.from_torch(torch.randn(64, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(ValueError, "rank"):
        groupnorm_sc_N_1_HW_C(x, 2)


def test_rejects_dim1_not_one(device, expect_error):
    x = ttnn.from_torch(torch.randn(1, 2, 64, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(ValueError, "dim"):
        groupnorm_sc_N_1_HW_C(x, 2)


def test_rejects_groups_not_dividing_channels(device, expect_error):
    x = ttnn.from_torch(torch.randn(1, 1, 64, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(ValueError, "num_groups"):
        groupnorm_sc_N_1_HW_C(x, 3)


def test_rejects_affine_shape_mismatch(device, expect_error):
    x = ttnn.from_torch(torch.randn(1, 1, 64, 64), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    g = ttnn.from_torch(torch.randn(1, 1, 1, 32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(ValueError, "gamma"):
        groupnorm_sc_N_1_HW_C(x, 2, gamma=g)
