# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Standalone SwiGLU activation correctness test for GPT-OSS MoE.

Tests the SwiGLU activation formula:
    gate_clamped = clamp(gate, max=7.0)
    up_clamped   = clamp(up, min=-7.0, max=7.0)
    result       = (up_clamped + 1) * gate_clamped * sigmoid(alpha * gate_clamped)

where alpha = 1.702.

This test validates:
1. The PyTorch reference implementation is correct
2. The TT device implementation matches the reference (via the full MOE op)
"""

import itertools
import math
import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------
def swiglu_reference(gate, up, alpha=1.702, clamp_limit=7.0):
    """
    PyTorch reference for GPT-OSS SwiGLU.

    Args:
        gate: Gate projection output (W0 @ input)
        up: Up projection output (W1 @ input)
        alpha: Sigmoid scaling factor (default: 1.702 for GPT-OSS)
        clamp_limit: Symmetric clamp bound (default: 7.0 for GPT-OSS)

    Returns:
        SwiGLU activated tensor
    """
    gate = torch.clamp(gate, max=clamp_limit)
    up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
    up_plus_1 = up + 1.0
    glu = gate * torch.sigmoid(alpha * gate)
    return up_plus_1 * glu


# ---------------------------------------------------------------------------
# Pure PyTorch reference tests (no hardware needed)
# ---------------------------------------------------------------------------
class TestSwiGLUReference:
    """Tests for the PyTorch SwiGLU reference implementation."""

    def test_basic_values(self):
        """SwiGLU with known values."""
        gate = torch.tensor([0.0, 1.0, -1.0, 0.5], dtype=torch.float32)
        up = torch.tensor([0.0, 1.0, -1.0, 0.5], dtype=torch.float32)
        result = swiglu_reference(gate, up)

        # gate=0: clamp(0)=0, sigmoid(0)=0.5, 0*0.5=0, (up+1)*0 = 0
        assert result[0].item() == pytest.approx(0.0, abs=1e-6)

        # gate=1: clamp(1)=1, sigmoid(1.702)~0.846, 1*0.846=0.846, (1+1)*0.846 = 1.692
        expected_1 = 2.0 * 1.0 * torch.sigmoid(torch.tensor(1.702)).item()
        assert result[1].item() == pytest.approx(expected_1, rel=1e-4)

    def test_clamping_gate(self):
        """Gate values above clamp_limit should be clamped."""
        gate = torch.tensor([10.0, 7.0, 6.9], dtype=torch.float32)
        up = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)  # up+1 = 1

        result = swiglu_reference(gate, up)

        # gate=10 clamped to 7, gate=7 stays, gate=6.9 stays
        # up=0 -> up+1 = 1 -> result = gate_clamped * sigmoid(alpha * gate_clamped)
        r_7 = 7.0 * torch.sigmoid(torch.tensor(1.702 * 7.0)).item()
        r_69 = 6.9 * torch.sigmoid(torch.tensor(1.702 * 6.9)).item()

        assert result[0].item() == pytest.approx(r_7, rel=1e-4), "gate=10 should be clamped to 7"
        assert result[1].item() == pytest.approx(r_7, rel=1e-4), "gate=7 should stay at 7"
        assert result[2].item() == pytest.approx(r_69, rel=1e-4), "gate=6.9 should stay at 6.9"

    def test_clamping_up(self):
        """Up values should be clamped symmetrically."""
        gate = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        up = torch.tensor([10.0, -10.0, 7.0, -7.0], dtype=torch.float32)

        result = swiglu_reference(gate, up)

        g = 1.0 * torch.sigmoid(torch.tensor(1.702)).item()  # gate part
        # up=10 clamped to 7 -> (7+1)*g = 8*g
        # up=-10 clamped to -7 -> (-7+1)*g = -6*g
        # up=7 -> (7+1)*g = 8*g
        # up=-7 -> (-7+1)*g = -6*g
        assert result[0].item() == pytest.approx(8.0 * g, rel=1e-4)
        assert result[1].item() == pytest.approx(-6.0 * g, rel=1e-4)
        assert result[2].item() == pytest.approx(8.0 * g, rel=1e-4)
        assert result[3].item() == pytest.approx(-6.0 * g, rel=1e-4)

    def test_bfloat16_range(self):
        """SwiGLU should work with bfloat16 typical values."""
        torch.manual_seed(42)
        gate = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
        up = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)

        result = swiglu_reference(gate.float(), up.float())
        result_bf16 = swiglu_reference(gate, up)

        # Check PCC between float32 and bfloat16 computation
        pcc = comp_pcc(result.bfloat16(), result_bf16)[0]
        assert pcc, f"BF16 PCC too low: {pcc}"

    def test_shape_preservation(self):
        """Output shape should match input shapes."""
        for shape in [(1, 1, 32, 32), (2, 4, 32, 64), (1, 1, 32, 2048)]:
            gate = torch.randn(shape, dtype=torch.float32)
            up = torch.randn(shape, dtype=torch.float32)
            result = swiglu_reference(gate, up)
            assert result.shape == gate.shape, f"Shape mismatch for {shape}"

    def test_vs_silu(self):
        """SwiGLU with alpha=1, no clamp, and up=0 should NOT equal SiLU (different formula)."""
        x = torch.randn(32, dtype=torch.float32)

        # SiLU: x * sigmoid(x)
        silu_result = torch.nn.functional.silu(x)

        # SwiGLU with alpha=1, no clamp, up=0: (0+1) * x * sigmoid(1*x) = x * sigmoid(x)
        swiglu_result = swiglu_reference(x, torch.zeros_like(x), alpha=1.0, clamp_limit=float("inf"))

        # These SHOULD be equal when alpha=1, up=0, no clamp
        assert torch.allclose(silu_result, swiglu_result, atol=1e-6), "SwiGLU(alpha=1, up=0) should equal SiLU"


# ---------------------------------------------------------------------------
# Integration test: run through the MOE op on device
# ---------------------------------------------------------------------------
# This imports the MOE test infrastructure and runs a minimal config
# to verify the SwiGLU activation on actual hardware.
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from test_moe import run_test_moe

PCC_THRESHOLD = 0.95  # Relaxed threshold for initial bring-up


@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW,
            },
            id="dispatch_row",
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "M, K, N, E, L",
    [(32, 7168, 2048, 2, 1)],
    ids=["M=32-K=7168-N=2048-E=2-L=1"],
)
def test_swiglu_via_moe(device, M, K, N, E, L):
    """
    Test SwiGLU activation by running it through the full MOE kernel.

    Uses the MOE test infrastructure with check_accuracy=True to verify
    that the SwiGLU output matches the PyTorch reference.
    """
    accuracy_metrics = run_test_moe(
        device,
        M,
        K,
        N,
        E,
        L,
        check_accuracy=True,
        dump_outputs=False,
    )

    for (layer_id, expert_id), metrics in accuracy_metrics.items():
        pcc_value = metrics.get("pcc", 0.0)
        logger.info(f"Layer {layer_id}, Expert {expert_id}: PCC = {pcc_value:.6f}")
        assert (
            pcc_value >= PCC_THRESHOLD
        ), f"PCC too low for layer {layer_id}, expert {expert_id}: {pcc_value:.6f} < {PCC_THRESHOLD}"
