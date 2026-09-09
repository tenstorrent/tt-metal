# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Numerical-stability / data-distribution / eps-variation regression tests
for groupnorm_sc_N_1_HW_C.

NOT driven by the registry. Runs unconditionally in the full suite
(subject to the op being callable). Tagged @pytest.mark.numerics. Covers
the cases the cross-product matrix in test_golden.py doesn't surface:
varying eps, varying input distribution, identity-affine sanity check.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from eval.golden_tests.groupnorm_sc_N_1_HW_C.helpers import (
    check_output,
    create_ttnn_input_tensor,
    pytorch_groupnorm_sc_N_1_HW_C,
)
from ttnn import migrated_run1000_groupnorm as groupnorm_sc_N_1_HW_C  # type: ignore


REPR_SHAPE = (1, 1, 64, 128)
REPR_NUM_GROUPS = 2
PCC_RELAXED = 0.990
RMS_RELAXED = 0.05


def _bf16_input(shape):
    torch.manual_seed(42)
    return torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)


def _bf16_weight(shape):
    torch.manual_seed(42)
    return torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)


def _run_default(device, torch_input, num_groups, *, gamma=None, beta=None, eps=1e-5, tolerance=None):
    """Run with the Phase-0 SUPPORTED cell (bf16 input + TILE + bf16 weights
    + ROW_MAJOR weights). `tolerance=(pcc, rms)` overrides the dtype default.
    """
    expected = pytorch_groupnorm_sc_N_1_HW_C(
        torch_input,
        num_groups,
        gamma=gamma,
        beta=beta,
        eps=eps,
    )
    ttnn_input = create_ttnn_input_tensor(
        torch_input,
        device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn_gamma = (
        create_ttnn_input_tensor(
            gamma,
            device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        if gamma is not None
        else None
    )
    ttnn_beta = (
        create_ttnn_input_tensor(
            beta,
            device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        if beta is not None
        else None
    )
    ttnn_output = groupnorm_sc_N_1_HW_C(
        ttnn_input,
        num_groups,
        gamma=ttnn_gamma,
        beta=ttnn_beta,
        eps=eps,
    )
    check_output(
        ttnn_output,
        expected,
        shape=list(torch_input.shape),
        dtype=ttnn.bfloat16,
        expected_layout=ttnn.TILE_LAYOUT,
        tolerance=tolerance,
    )


# --- 1. Eps variation ----------------------------------------------------

EPS_VALUES = [
    pytest.param(1e-5, id="eps_1e-5"),
    pytest.param(1e-6, id="eps_1e-6"),
    pytest.param(1e-3, id="eps_1e-3"),
    pytest.param(1e-2, id="eps_1e-2"),
]


@pytest.mark.numerics
@pytest.mark.parametrize("eps", EPS_VALUES)
def test_eps_variation(device, eps):
    torch_input = _bf16_input(REPR_SHAPE)
    gamma = _bf16_weight((1, 1, 1, REPR_SHAPE[-1]))
    beta = _bf16_weight((1, 1, 1, REPR_SHAPE[-1]))
    _run_default(device, torch_input, REPR_NUM_GROUPS, gamma=gamma, beta=beta, eps=eps)


# --- 2. Identity affine (gamma=1, beta=0) --------------------------------


@pytest.mark.numerics
def test_identity_affine(device):
    """Identity affine: output should equal normalized input."""
    C = REPR_SHAPE[-1]
    torch_input = _bf16_input(REPR_SHAPE)
    gamma = torch.ones(1, 1, 1, C, dtype=torch.bfloat16)
    beta = torch.zeros(1, 1, 1, C, dtype=torch.bfloat16)
    _run_default(device, torch_input, REPR_NUM_GROUPS, gamma=gamma, beta=beta)


# --- 3. Data distribution variations -------------------------------------

DIST_SHAPES = [
    (1, 1, 32, 32),
    (1, 1, 128, 256),
    (1, 1, 64, 512),
    (2, 1, 64, 128),
    (1, 1, 512, 128),
]


def _randn(shape):
    return torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)


def _uniform(shape):
    return torch.rand(shape, dtype=torch.bfloat16)


def _small(shape):
    return torch.randn(shape, dtype=torch.float32).to(torch.bfloat16) * 0.01


def _large(shape):
    return torch.randn(shape, dtype=torch.float32).to(torch.bfloat16) * 10.0


def _positive(shape):
    return torch.rand(shape, dtype=torch.bfloat16) + 0.5


def _negative(shape):
    return -(torch.rand(shape, dtype=torch.bfloat16) + 0.5)


DISTRIBUTIONS = [
    pytest.param(_randn, id="randn"),
    pytest.param(_uniform, id="uniform"),
    pytest.param(_small, id="small"),
    pytest.param(_large, id="large"),
    pytest.param(_positive, id="positive"),
    pytest.param(_negative, id="negative"),
]


@pytest.mark.numerics
@pytest.mark.parametrize("make_input", DISTRIBUTIONS)
@pytest.mark.parametrize(
    "shape",
    DIST_SHAPES,
    ids=[f"{s[0]}x{s[1]}x{s[2]}x{s[3]}" for s in DIST_SHAPES],
)
def test_distributions(device, shape, make_input):
    """Cover near-zero / large-magnitude / single-sign distributions.

    num_groups=1 (group-norm degenerates to layer-norm-over-spatial). Small
    inputs get relaxed tolerance — the variance collapses and bf16
    precision is barely sufficient.
    """
    torch.manual_seed(42)
    C = shape[-1]
    torch_input = make_input(shape)
    gamma = _bf16_weight((1, 1, 1, C))
    beta = _bf16_weight((1, 1, 1, C))

    tolerance = (PCC_RELAXED, RMS_RELAXED) if make_input is _small else None
    _run_default(device, torch_input, 1, gamma=gamma, beta=beta, tolerance=tolerance)
