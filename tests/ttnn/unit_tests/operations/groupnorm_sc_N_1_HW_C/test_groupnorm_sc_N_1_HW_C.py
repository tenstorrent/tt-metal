# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for groupnorm_sc_N_1_HW_C (single-core GroupNorm, (N, 1, H*W, C) layout).

This file is the immutable spec for the implementer — do NOT modify it.

Reference: torch.nn.functional.group_norm on the (N, C, HW) permutation of the
input. Output contract: shape == input shape, dtype == input dtype, layout is
ALWAYS TILE_LAYOUT regardless of input layout.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
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


def torch_groupnorm(x, num_groups, gamma=None, beta=None, eps=1e-5):
    """GroupNorm reference for (N, 1, HW, C) layout, computed in fp32."""
    N, one, HW, C = x.shape
    x_nchw = x.to(torch.float32).squeeze(1).permute(0, 2, 1)  # (N, C, HW)
    weight = gamma.to(torch.float32).reshape(C) if gamma is not None else None
    bias = beta.to(torch.float32).reshape(C) if beta is not None else None
    y = torch.nn.functional.group_norm(x_nchw, num_groups, weight=weight, bias=bias, eps=eps)
    return y.permute(0, 2, 1).unsqueeze(1)  # (N, 1, HW, C)


def run_case(device, shape, num_groups, layout, affine, dtype=ttnn.bfloat16, eps=1e-5):
    torch.manual_seed(42)
    N, _, HW, C = shape
    tdtype = TORCH_DTYPE[dtype]

    torch_input = torch.randn(shape, dtype=tdtype)
    gamma = torch.randn((1, 1, 1, C), dtype=torch.bfloat16) if affine in ("gamma_beta", "gamma_only") else None
    beta = torch.randn((1, 1, 1, C), dtype=torch.bfloat16) if affine == "gamma_beta" else None

    expected = torch_groupnorm(torch_input, num_groups, gamma, beta, eps)

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device)
    kwargs = {"eps": eps}
    if gamma is not None:
        kwargs["gamma"] = ttnn.from_torch(gamma, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    if beta is not None:
        kwargs["beta"] = ttnn.from_torch(beta, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    tt_output = groupnorm_sc_N_1_HW_C(tt_input, num_groups, **kwargs)

    assert tt_output.layout == ttnn.TILE_LAYOUT, "output layout must always be TILE_LAYOUT"
    assert tt_output.dtype == tt_input.dtype, "output dtype must equal input dtype"
    assert list(tt_output.shape) == list(shape), "output shape must equal input shape"

    result = ttnn.to_torch(tt_output).to(torch.float32)
    assert_with_pcc(expected.to(torch.float32), result, pcc=PCC_BY_DTYPE[dtype])


# --- shapes: single-tile, multi-tile, non-square, multi-batch, odd group count ---
SHAPES = [
    pytest.param((1, 1, 32, 32), 1, id="single_tile_g1"),
    pytest.param((1, 1, 64, 128), 4, id="multi_tile_g4"),
    pytest.param((1, 1, 256, 64), 2, id="tall_nonsquare_g2"),
    pytest.param((1, 1, 32, 512), 16, id="wide_nonsquare_g16"),
    pytest.param((2, 1, 64, 128), 4, id="multi_batch_g4"),
    pytest.param((1, 1, 64, 96), 3, id="odd_groups_g3"),
]


@pytest.mark.parametrize("shape, num_groups", SHAPES)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
def test_groupnorm_full_affine(device, shape, num_groups, layout):
    run_case(device, shape, num_groups, layout, affine="gamma_beta")


@pytest.mark.parametrize("shape, num_groups", SHAPES)
def test_groupnorm_no_affine(device, shape, num_groups):
    run_case(device, shape, num_groups, ttnn.TILE_LAYOUT, affine="no_affine")


@pytest.mark.parametrize("shape, num_groups", SHAPES)
def test_groupnorm_gamma_only(device, shape, num_groups):
    run_case(device, shape, num_groups, ttnn.TILE_LAYOUT, affine="gamma_only")


@pytest.mark.parametrize("eps", [1e-6, 1e-3])
def test_groupnorm_custom_eps(device, eps):
    run_case(device, (1, 1, 64, 128), 4, ttnn.TILE_LAYOUT, affine="gamma_beta", eps=eps)


def test_groupnorm_large_group(device):
    # one big group over a 512x256 slab (stress for streaming stats)
    run_case(device, (1, 1, 512, 256), 1, ttnn.TILE_LAYOUT, affine="gamma_beta")


# --- argument validation (ValueError, not NotImplementedError) ---


def test_rejects_bad_rank(device):
    t = ttnn.from_torch(
        torch.randn(32, 32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    with pytest.raises(ValueError):
        groupnorm_sc_N_1_HW_C(t, 1)


def test_rejects_bad_dim1(device):
    t = ttnn.from_torch(
        torch.randn(1, 2, 32, 32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    with pytest.raises(ValueError):
        groupnorm_sc_N_1_HW_C(t, 1)


def test_rejects_indivisible_groups(device):
    t = ttnn.from_torch(
        torch.randn(1, 1, 32, 96, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    with pytest.raises(ValueError):
        groupnorm_sc_N_1_HW_C(t, 5)  # 96 % 5 != 0


def test_rejects_bad_gamma_shape(device):
    t = ttnn.from_torch(
        torch.randn(1, 1, 32, 64, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    g = ttnn.from_torch(
        torch.randn(1, 1, 1, 32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    with pytest.raises(ValueError):
        groupnorm_sc_N_1_HW_C(t, 2, gamma=g)


def test_group_not_tile_aligned_is_refinement_gate(device):
    # (C/G) % 32 != 0 — kernel-side constraint, gated as NotImplementedError
    t = ttnn.from_torch(
        torch.randn(1, 1, 32, 320, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    with pytest.raises(NotImplementedError):
        groupnorm_sc_N_1_HW_C(t, 32)  # Cg = 10
