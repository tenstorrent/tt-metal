# SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for gated_delta_net (Phase 0: float32, HiFi4, fp32_dest_acc_en=True).

Measures PCC, abs errors, relative RMS, ULP p99 and the got/true ratio spread of `output` and
`recurrent_state_out`; gates on PCC and relative RMS together, as eval.metrics.check_output does.
"""

import pytest
import torch

import ttnn
from eval.metrics import compute_metrics_torch, default_tolerance
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.gated_delta_net import gated_delta_net
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import (
    ORDER,
    compute_config,
    make_inputs,
    reference,
    to_device,
)

SHAPES = [
    pytest.param(((1, 1, 256), (1, 3, 32, 64)), id="decode_b1_small"),
    pytest.param(((1, 1, 2560), (1, 12, 128, 128)), id="decode_b1_chip"),
    pytest.param(((1, 33, 256), (1, 3, 32, 64)), id="prefill_t33_small"),
    pytest.param(((1, 512, 256), (1, 3, 32, 64)), id="prefill_t512_small"),
    pytest.param(((1, 64, 2560), (1, 12, 128, 128)), id="prefill_t64_chip"),
]


def _ratio_spread(actual, expected):
    """Median and p5/p95 of actual/expected where the reference is clearly non-zero."""
    mask = torch.isfinite(actual) & (expected.abs() > 1e-3 * expected.abs().max())
    ratio = (actual[mask] / expected[mask]).double()
    q = torch.quantile(ratio[: 1 << 20], torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))
    return q[1].item(), q[0].item(), q[2].item()


def _measure(label, actual, expected):
    actual = actual.to(torch.float32)
    expected = expected.to(torch.float32)
    _, pcc = comp_pcc(expected, actual)
    _, allclose = comp_allclose(expected, actual)
    metrics = compute_metrics_torch(actual, expected, readback_dtype=torch.float32)
    mean_abs = (actual - expected).abs().mean().item()
    median, p5, p95 = _ratio_spread(actual.flatten(), expected.flatten())
    print(
        f"\n{label}: PCC={pcc:.8f} max_abs={metrics.max_abs_diff:.3e} mean_abs={mean_abs:.3e} "
        f"rel_rms={metrics.rms:.3e} ulp_p99={metrics.ulp_p99:.0f} "
        f"ratio median={median:.6f} p5={p5:.6f} p95={p95:.6f} ({allclose})"
    )
    return pcc, metrics.rms, median


@pytest.mark.parametrize("inputs", SHAPES)
def test_precision_baseline(device, inputs):
    torch.manual_seed(0)
    tensors = make_inputs(inputs)
    expected = reference(*(tensors[name] for name in ORDER))
    out, _, state = gated_delta_net(*to_device(tensors, device), compute_kernel_config=compute_config())
    pcc_min, rms_max = default_tolerance(ttnn.float32, True)
    for label, actual, ref in (
        ("output", ttnn.to_torch(out), expected[0]),
        ("recurrent_state_out", ttnn.to_torch(state), expected[2]),
    ):
        pcc, rel_rms, median = _measure(label, actual, ref)
        assert pcc >= pcc_min, f"{label}: PCC {pcc} < {pcc_min}"
        assert rel_rms <= rms_max, f"{label}: relative RMS {rel_rms} > {rms_max}"
        # Prefill states sit at a median of ~0.999 with a broad p5..p95 spread: the 19-bit srcA/srcB
        # truncation of the increment operands biases toward zero (allowed by the spec). A uniform
        # scale bug shows as a tight cluster far from 1.
        assert abs(median - 1.0) < 5e-3, f"{label}: got/true ratio median {median} (scale bug)"
