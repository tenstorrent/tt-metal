# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tanh Backward FP32-destination Precision Tests

Companion to test_tanh_bw_ulp.py, which covers the bfloat16-destination path.
A float32 input tensor makes tanh_bw_program_factory set fp32_dest_acc_en, which
instantiates calculate_tanh_derivative_sech2<..., is_fp32_dest_acc_en = true>.
The reference for that path is the same sech²(x) = 1/cosh²(x) evaluated at 256-bit
precision, but the tolerance is an fp32 ULP rather than a bfloat16 one.

The kernel's exp tail flushes a subnormal result to zero (SFPU is FTZ), so the
reference does the same: below the fp32 minimum normal the expected value is 0.

Run: pytest tests/ttnn/unit_tests/operations/eltwise/test_tanh_bw_fp32_ulp.py -v -s
"""

import struct

import numpy as np
import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import sech2_exact

FP32_MIN_NORMAL = float(np.finfo(np.float32).tiny)  # 1.1754944e-38

# The exact identity sech²(x) = 4e/(1+e)², e = exp(-2|x|), costs one rounding for the
# exp, one for (1+e)², one for the reciprocal and one for the final multiply.
FP32_MAX_ULP = 4

# The fp32 path is Blackhole-only for now: the Wormhole copy of
# ckernel_sfpu_tanh_derivative.h still carries the bfloat16-grade approximation on
# its fp32-dest instantiation (see tenstorrent/tt-metal#57509, defect 3).
pytestmark = pytest.mark.skipif(
    ttnn.get_arch_name() != "blackhole",
    reason=(
        "fp32-dest tanh derivative is only fixed on Blackhole; Wormhole tracked by "
        "https://github.com/tenstorrent/tt-metal/issues/57509"
    ),
)


def sech2_exact_fp32_input(x: float) -> float:
    """sech²(x) at 256-bit precision, evaluated at the fp32-rounded input.

    Rounding the argument first matters: sech² has relative derivative -2, so a
    half-ulp perturbation of a large x is worth tens of ulp in the result, and
    would otherwise be charged to the kernel.
    """
    result = sech2_exact(float(np.float32(x)))
    return 0.0 if result < FP32_MIN_NORMAL else float(np.float32(result))


def _fp32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", np.float32(x)))[0]


def ulp_distance_fp32(actual: float, expected: float) -> int:
    """Distance in fp32 ULP. The bit difference assumes both arguments share a sign."""
    a, e = np.float32(actual), np.float32(expected)
    if np.isnan(a) or np.isnan(e):
        return 0 if (np.isnan(a) and np.isnan(e)) else 2**31
    if float(a) == float(e):
        return 0
    return abs(_fp32_bits(a) - _fp32_bits(e))


def _run_tanh_bw_fp32(device, xs, grads=None) -> np.ndarray:
    """Run ttnn.tanh_bw over a list of float32 inputs in one device call."""
    if grads is None:
        grads = [1.0] * len(xs)
    torch_input = torch.tensor([xs], dtype=torch.float32)
    torch_grad = torch.tensor([grads], dtype=torch.float32)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, device=device, layout=ttnn.TILE_LAYOUT)
    tt_grad = ttnn.from_torch(torch_grad, dtype=ttnn.float32, device=device, layout=ttnn.TILE_LAYOUT)

    results = ttnn.tanh_bw(tt_grad, tt_input)
    return ttnn.to_torch(results[0]).to(torch.float32).numpy().reshape(-1)


def _report(xs, actual, max_report=8):
    """Log the worst offenders and return (max_ulp, worst_x)."""
    rows = []
    for x, a in zip(xs, actual):
        e = sech2_exact_fp32_input(x)
        rows.append((ulp_distance_fp32(a, e), x, float(a), e))
    rows.sort(reverse=True)
    logger.info(f"{'x':>12} | {'expected':>16} | {'actual':>16} | {'fp32 ULP':>10}")
    for u, x, a, e in rows[:max_report]:
        logger.info(f"{x:>12.6g} | {e:>16.9e} | {a:>16.9e} | {u:>10}")
    return rows[0][0], rows[0][1]


# =============================================================================
# Regression tests
# =============================================================================


class TestTanhBwFp32Peak:
    """sech²(0) = 1 exactly. The bfloat16-grade degree-10 polynomial evaluates its
    scaled variable at t = -1 here and lands on 0.999197, which rounds to 1.0 in
    bfloat16 but is ~13k ULP away in fp32."""

    def test_derivative_at_zero(self, device):
        actual = _run_tanh_bw_fp32(device, [0.0])[0]
        ulp = ulp_distance_fp32(actual, 1.0)
        logger.info(f"x=0: expected=1.0, actual={actual:.9f}, fp32 ULP={ulp}")
        assert ulp <= FP32_MAX_ULP, f"sech2(0) should be 1.0 within {FP32_MAX_ULP} fp32 ULP, got {actual!r} ({ulp} ULP)"


class TestTanhBwFp32Monotonicity:
    """sech² is strictly decreasing in |x|. The bfloat16-grade math this arm replaced
    switched approximation at |x| = 3, and the mismatch between its two pieces
    stepped the function back up across that boundary. The fp32 arm is now one
    formula with no boundary; these cases guard against that step coming back."""

    def test_no_step_across_core_tail_boundary(self, device):
        xs = [2.999, 2.9999, 3.0, 3.0001, 3.001]
        actual = _run_tanh_bw_fp32(device, xs)
        for x, a in zip(xs, actual):
            logger.info(f"x={x:<8} actual={a:.9e} expected={sech2_exact_fp32_input(x):.9e}")
        for i in range(len(xs) - 1):
            assert actual[i] > actual[i + 1], (
                f"sech2 must decrease across |x|=3 (the old bf16-grade region boundary), but "
                f"f({xs[i]})={actual[i]:.9e} <= f({xs[i + 1]})={actual[i + 1]:.9e}"
            )

    def test_monotone_decreasing_over_range(self, device):
        xs = list(np.linspace(0.0, 44.0, 512, dtype=np.float32))
        actual = _run_tanh_bw_fp32(device, [float(x) for x in xs])
        for i in range(len(xs) - 1):
            assert actual[i] >= actual[i + 1], (
                f"sech2 must be non-increasing in |x|, but "
                f"f({xs[i]})={actual[i]:.9e} < f({xs[i + 1]})={actual[i + 1]:.9e}"
            )


class TestTanhBwFp32Accuracy:
    """Per-region fp32 ULP, against FP32_MAX_ULP."""

    @pytest.mark.parametrize(
        "name,xs",
        [
            ("near-zero", [0.0, 1e-8, 1e-6, 1e-4, 0.001, 0.01, 0.1]),
            ("core", [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 2.9]),
            ("boundary", [2.99, 2.999, 3.0, 3.001, 3.01, 3.1]),
            ("tail", [3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 40.0, 43.0, 44.0]),
            ("negative", [-0.0, -0.5, -1.0, -2.0, -2.999, -3.0, -5.0, -20.0, -44.0]),
        ],
    )
    def test_ulp_by_region(self, device, name, xs):
        actual = _run_tanh_bw_fp32(device, xs)
        max_ulp, worst_x = _report(xs, actual)
        assert max_ulp <= FP32_MAX_ULP, f"{name}: max fp32 ULP {max_ulp} at x={worst_x} exceeds {FP32_MAX_ULP}"

    def test_ulp_sweep(self, device):
        """Dense deterministic sweep over the whole non-saturated range."""
        rng = np.random.default_rng(20260924)
        xs = [float(v) for v in np.concatenate([rng.uniform(0.0, 3.0, 512), rng.uniform(3.0, 44.0, 512)])]
        actual = _run_tanh_bw_fp32(device, xs)
        max_ulp, worst_x = _report(xs, actual)
        assert max_ulp <= FP32_MAX_ULP, f"sweep: max fp32 ULP {max_ulp} at x={worst_x} exceeds {FP32_MAX_ULP}"


class TestTanhBwFp32Saturation:
    """FTZ behaviour at the far end must survive the fp32 retune: 4·exp(-2|x|) is
    still a normal fp32 number out to |x| ~= 44.4, and zero past it. The infinities
    and NaN of either sign also return 0; -NaN is the case that sfpi::abs would have
    let through, since it leaves the sign bit of a NaN set."""

    def test_saturation_boundary(self, device):
        xs = [
            43.0,
            43.75,
            44.0,
            44.3,
            44.5,
            45.0,
            50.0,
            100.0,
            float("inf"),
            -float("inf"),
            float("nan"),
            -float("nan"),
        ]
        actual = _run_tanh_bw_fp32(device, xs)
        for x, a in zip(xs, actual):
            expected = 0.0 if not np.isfinite(x) else sech2_exact_fp32_input(x)
            logger.info(f"x={x:<8} expected={expected:.9e} actual={float(a):.9e}")
            if expected == 0.0:
                assert float(a) == 0.0, f"expected flush to zero at x={x}, got {float(a)!r}"
            else:
                assert ulp_distance_fp32(a, expected) <= FP32_MAX_ULP, f"x={x}: {float(a)!r} vs {expected!r}"


class TestTanhBwFp32Gradient:
    """grad * sech²(x): the multiply is host-side exact for these grads, so the
    tolerance stays the kernel's."""

    @pytest.mark.parametrize("x,grad", [(0.0, 2.0), (1.0, 0.5), (-1.0, 2.0), (2.0, -1.0), (5.0, 4.0)])
    def test_with_gradient(self, device, x, grad):
        actual = _run_tanh_bw_fp32(device, [x], [grad])[0]
        expected = np.float32(np.float32(grad) * np.float32(sech2_exact_fp32_input(x)))
        ulp = ulp_distance_fp32(actual, expected)
        logger.info(f"x={x}, grad={grad}: expected={float(expected):.9e}, actual={float(actual):.9e}, ULP={ulp}")
        assert ulp <= FP32_MAX_ULP, f"x={x}, grad={grad}: {float(actual)!r} vs {float(expected)!r} ({ulp} ULP)"
