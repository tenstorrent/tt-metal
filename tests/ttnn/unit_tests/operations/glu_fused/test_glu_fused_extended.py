# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Extended tests for glu_fused — focused coverage beyond the acceptance suite.

The acceptance test (test_glu_fused.py) already covers correctness across
representative shapes and the negative-case validation matrix. This file
adds a small, targeted set:

1. **L1 memory_config preservation** — output should inherit input's L1
   memory config, not silently fall back to DRAM. The acceptance test only
   exercises DRAM inputs.
2. **Deterministic ascending-value structural check on a wide W** — exercises
   the reader's split-offset arithmetic on a shape with many output tiles per
   row (W=512 → 8 output tiles per tile-row).
3. **Sigmoid edge behavior** — large-magnitude inputs in the B half should
   saturate sigmoid to ~0 or ~1, and the multiply should produce the A half
   (when B ≫ 0) or near-zero (when B ≪ 0). Validates the accurate-sigmoid
   path under saturation.
4. **Determinism** — running the same op twice on the same input should
   produce bit-identical outputs.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.glu_fused import glu_fused


def _glu_ref(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.glu(x, dim=-1)


# -----------------------------------------------------------------------------
# L1 memory config — output should inherit
# -----------------------------------------------------------------------------


def test_glu_fused_l1_memory_config_preserved(device):
    """Input in L1 should produce an output in L1 (memory_config inheritance)."""
    shape = (1, 1, 32, 128)
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.float32)
    torch_expected = _glu_ref(torch_input)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    ttnn_output = glu_fused(ttnn_input)
    assert ttnn_output.memory_config() == ttnn_input.memory_config(), (
        f"L1 memory config not preserved: in={ttnn_input.memory_config()} " f"out={ttnn_output.memory_config()}"
    )
    assert_with_pcc(torch_expected, ttnn.to_torch(ttnn_output).float(), pcc=0.999)


# -----------------------------------------------------------------------------
# Wide-W structural check
# -----------------------------------------------------------------------------


def test_glu_fused_wide_W_arange(device):
    """
    Deterministic ascending input on W=512 — 8 output tiles per tile-row.
    Off-by-one in the per-iter split offset would show up at recognizable
    column positions.
    """
    shape = (1, 1, 64, 512)
    n = shape[0] * shape[1] * shape[2] * shape[3]
    # Normalize to keep sigmoid in the well-behaved range.
    torch_input = (torch.arange(n, dtype=torch.float32) / n - 0.5).reshape(shape)
    torch_expected = _glu_ref(torch_input)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    actual = ttnn.to_torch(glu_fused(ttnn_input)).float()
    assert_with_pcc(torch_expected, actual, pcc=0.999)


# -----------------------------------------------------------------------------
# Sigmoid saturation
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "b_val,sigmoid_target",
    [
        pytest.param(-20.0, 0.0, id="b_very_negative_saturates_low"),
        pytest.param(20.0, 1.0, id="b_very_positive_saturates_high"),
    ],
)
def test_glu_fused_sigmoid_saturation(device, b_val, sigmoid_target):
    """
    With B-half far from 0, sigmoid(B) saturates to ~0 or ~1. Output should
    be approximately A * sigmoid_target. Confirms accurate sigmoid (not
    fast-approx) handles saturation correctly.
    """
    shape = (1, 1, 32, 64)  # halves of 32 each
    torch.manual_seed(7)
    a = torch.randn((1, 1, 32, 32), dtype=torch.float32)
    b = torch.full((1, 1, 32, 32), b_val, dtype=torch.float32)
    torch_input = torch.cat([a, b], dim=-1)
    torch_expected = _glu_ref(torch_input)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    actual = ttnn.to_torch(glu_fused(ttnn_input)).float()

    # Compare against torch's own glu (it also uses accurate sigmoid).
    assert_with_pcc(torch_expected, actual, pcc=0.999)

    # And against the saturation expectation directly.
    expected_saturated = a * sigmoid_target
    max_err = (actual - expected_saturated).abs().max().item()
    assert max_err <= 1e-3, (
        f"Saturation expectation violated: max abs err = {max_err:.6e} "
        f"for b_val={b_val} (sigmoid_target={sigmoid_target})"
    )


# -----------------------------------------------------------------------------
# Determinism
# -----------------------------------------------------------------------------


def test_glu_fused_determinism(device):
    """Running the op twice on the same input must produce identical results."""
    shape = (2, 2, 64, 128)
    torch.manual_seed(123)
    torch_input = torch.randn(shape, dtype=torch.float32)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out_a = ttnn.to_torch(glu_fused(ttnn_input)).float()
    out_b = ttnn.to_torch(glu_fused(ttnn_input)).float()
    assert torch.equal(out_a, out_b), (
        f"glu_fused is non-deterministic: max diff = " f"{(out_a - out_b).abs().max().item():.6e}"
    )
