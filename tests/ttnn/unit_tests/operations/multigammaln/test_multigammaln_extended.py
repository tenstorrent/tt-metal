# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Extended tests for multigammaln (order p = 4).

The acceptance test covers shapes, the reflection branch, NaN propagation, and
validation. This file fills in a small set of focused gaps the verifier wants
green at Phase 0:

  - L1 output memory config (vs DRAM) — proves the entry point honors
    `memory_config=`.
  - "Large" magnitudes where Stirling dominates and the reflection path is
    fully cold (a >> 2).
  - Domain boundary spread: a just above 1.5 (every lgamma argument > 0 by a
    hair); checks the reflection path delivers finite results across an entire
    tile, not just a few sampled values.
  - A larger multi-core shape (a few hundred tiles) — catches subtle work-
    distribution bugs that single-tile or 4-tile cases would miss.

Test matrix is deliberately small. Broader sweeps belong in refinements.
"""

import math

import pytest
import torch
import ttnn

from ttnn.operations.multigammaln import multigammaln


RTOL = 0.05
ATOL = 0.2


def _torch_reference(a: torch.Tensor) -> torch.Tensor:
    return torch.special.multigammaln(a.float(), 4)


@pytest.mark.parametrize(
    "memory_config",
    [
        pytest.param(ttnn.DRAM_MEMORY_CONFIG, id="dram_out"),
        pytest.param(ttnn.L1_MEMORY_CONFIG, id="l1_out"),
    ],
)
def test_multigammaln_output_memory_config(device, memory_config):
    """Entry point must honor the `memory_config=` kwarg for the output."""
    torch.manual_seed(0)
    shape = (1, 1, 64, 64)
    torch_input = 3.0 + 2.0 * torch.rand(shape, dtype=torch.float32)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output = multigammaln(ttnn_input, memory_config=memory_config)
    actual = ttnn.to_torch(ttnn_output).float()
    expected = _torch_reference(torch_input)

    assert torch.allclose(actual, expected, rtol=RTOL, atol=ATOL)


def test_multigammaln_large_magnitude_stirling(device):
    """
    For large a, every lgamma argument is large positive and the kernel
    stays exclusively on the Stirling path (no reflection). Output grows
    quickly — verify relative tolerance still holds.
    """
    torch.manual_seed(7)
    shape = (1, 1, 32, 256)
    torch_input = 20.0 + 30.0 * torch.rand(shape, dtype=torch.float32)  # a in [20, 50]

    torch_expected = _torch_reference(torch_input)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output = multigammaln(ttnn_input)
    actual = ttnn.to_torch(ttnn_output).float()

    # Relative tolerance: at a=50 the output is around 4*lgamma(50) ≈ 4*144 = 576
    # plus 3*log(pi), so absolute error of ~0.2 is far below 1e-4 relative.
    assert torch.isfinite(actual).all()
    max_rel = ((actual - torch_expected).abs() / torch_expected.abs().clamp(min=1.0)).max().item()
    assert max_rel < 1e-3, f"Stirling-domain relative error too large: {max_rel:.4g}"


def test_multigammaln_domain_boundary_strip(device):
    """
    Cover a fine grid of in-domain values just above the boundary (a in
    [1.6, 2.5]). Every tile of the output must be finite; every lgamma
    argument is positive (a - 1.5 >= 0.1), and the reflection branch fires
    for the (a - 1.0) and (a - 1.5) terms.
    """
    torch.manual_seed(13)
    shape = (1, 1, 64, 64)
    torch_input = 1.6 + 0.9 * torch.rand(shape, dtype=torch.float32)  # (1.6, 2.5)

    torch_expected = _torch_reference(torch_input)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output = multigammaln(ttnn_input)
    actual = ttnn.to_torch(ttnn_output).float()

    assert torch.isfinite(actual).all(), "Reflection-branch strip produced non-finite values"
    assert torch.allclose(actual, torch_expected, rtol=RTOL, atol=ATOL)


def test_multigammaln_multi_core_distribution(device):
    """
    Larger shape (several hundred tiles) — exercises `split_work_to_cores`'s
    two-group split. If the per-core RT-arg fan-out is wrong (e.g., wrong
    start_tile_id assignment), the failure tends to manifest as a "wrong
    answer in a contiguous chunk of the output" — easily caught here.
    """
    torch.manual_seed(2026)
    # 16x16 = 256 output tiles → with a 7x8=56-core Wormhole grid the split
    # is roughly 4 or 5 tiles per core, splitting across both groups.
    shape = (1, 1, 512, 512)
    torch_input = 2.5 + 4.0 * torch.rand(shape, dtype=torch.float32)

    torch_expected = _torch_reference(torch_input)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output = multigammaln(ttnn_input)
    actual = ttnn.to_torch(ttnn_output).float()

    assert torch.allclose(actual, torch_expected, rtol=RTOL, atol=ATOL)
