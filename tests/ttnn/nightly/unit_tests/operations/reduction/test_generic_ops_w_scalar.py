# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Generic reductions with the scalar= post-scale argument.
# Split from test_reduction_ops.py: corner-case tests, not exhaustive sweeps.

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_numeric_metrics
from tests.ttnn.nightly.unit_tests.operations.reduction.utility_functions import (
    TTNN_REDUCTION_WRAPPERS,
)

# Module-scoped device: these tests all run with the default device config, so the
# device is opened once per file (one device context per test group) instead of
# once per test case.
pytestmark = pytest.mark.use_module_device


SHAPES = [(3, 4), (1, 1, 3, 4, 5), (3, 4, 8, 56, 33)]
DIMS_PART1 = [(-2, -1), None]
DIMS_PART2 = [-1, -2, 0, (-2, -1), (0, -2, -1), None]


# Pair (shape, dim) to drop combos with more reduction dims than rank, and dims that
# duplicate another dim of the same list at rank 2.
def _valid_shape_dims(dims, rank2_duplicates):
    return [
        (s, d)
        for s in SHAPES
        for d in dims
        if not (isinstance(d, tuple) and len(d) > len(s)) and not (len(s) == 2 and d in rank2_duplicates)
    ]


# At rank 2, dim=(-2, -1) is the same reduction as dim=None, and dim=0 the same as dim=-2.
VALID_SHAPE_DIMS_PART1 = _valid_shape_dims(DIMS_PART1, rank2_duplicates=[None])
VALID_SHAPE_DIMS_PART2 = _valid_shape_dims(DIMS_PART2, rank2_duplicates=[0, (-2, -1)])

# Pair (op, correction) since correction is live only for std/var.
OP_CORRECTION = [
    ("sum", False),
    ("mean", False),
    ("max", False),
    ("min", False),
    ("std", False),
    ("std", True),
    ("var", False),
    ("var", True),
]


def _run_generic_ops_w_scalar(device, op, scalar, correction, dim, shape, dtype):
    if op in ("var", "std") and correction:
        # Bessel's correction divides by (N - 1) where N is the number of elements
        # reduced. With N == 1 this is a divide-by-zero, producing all-NaN output;
        # both PyTorch and ttnn match here, but the result is mathematically
        # degenerate and not meaningful to compare numerically.
        if dim is None:
            reduction_count = 1
            for s in shape:
                reduction_count *= s
        elif isinstance(dim, tuple):
            reduction_count = 1
            for d in dim:
                reduction_count *= shape[d]
        else:
            reduction_count = shape[dim]
        if reduction_count <= 1:
            pytest.skip("Bessel-corrected std/var with reduction count of 1 produces NaN output")

    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=dtype)

    ttnn_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_op = TTNN_REDUCTION_WRAPPERS[op]
    ttnn_result = ttnn.to_torch(ttnn_op(ttnn_input, dim=dim, scalar=scalar, correction=correction))

    # torch.max/min don't accept a tuple for dim; use amax/amin which do.
    torch_op_name = {"max": "amax", "min": "amin"}.get(op, op)
    torch_op = getattr(torch, torch_op_name)
    # PyTorch supports the correction argument only for var and std.
    # ttnn supports it for all, but it is ignored for all except var and std.
    if op in ("var", "std"):
        torch_result = torch_op(scalar * torch_input, dim=dim, correction=correction)
    else:
        torch_result = torch_op(scalar * torch_input, dim=dim)

    if dtype == torch.float32 and op in ("var", "std"):
        # FP32 var/std use shifted two-pass SFPU statistics without pre-scaling the input. The scalar
        # is applied afterwards (var(s*x) = s^2 var(x), std(s*x) = |s| std(x)), preserving the full
        # FP32 input and accumulation precision while avoiding the FPU scale-multiply. For the largest
        # case here, rtol = 1e-4 covers accumulated rounding with margin and atol = 1e-4 handles the
        # small-magnitude regime.
        rtol = 1e-4
        atol = 1e-4
        frobenius_threshold = 1e-4
        pcc = 0.9999
    elif dtype == torch.float32:
        # sum/mean/max/min on FP32 go through the FPU SrcA TF32 floor: inputs
        # are loaded at TF32 precision (10-bit mantissa, eps_TF32 = 2^-10 ~= 1e-3)
        # before any arithmetic. This is the precision floor; FP32 accumulation
        # after the truncated load does not recover it.
        #
        # Notation used in the comments below:
        #   x_i = an individual input element (after the implicit scalar*input
        #         applied by the ttnn op); the test feeds randn samples so
        #         |x_i| has unit stddev pre-scaling and |scalar| stddev post-scaling.
        #   y, y_i = the reduction output (scalar for dim=None, otherwise an
        #            element of the output tensor).
        #   N = number of input elements reduced into a single output (the full
        #       input count for dim=None, the product of the reduced axes for
        #       a tuple, the single axis size for an int dim).
        #
        # Per-element error model:
        # 13 mantissa bits are discarded by FP32->TF32, so for any
        # x_i the truncated value has magnitude <= |x_i|. Treating the dropped
        # fraction as uniform on [0, eps_TF32 * |x_i|] gives:
        #   E[error_i] = -eps_TF32 / 2 * x_i   (sign(error_i) = -sign(x_i))
        #   Var(error_i) = (eps_TF32 * |x_i|)^2 / 12   (uniform on a 1-ULP interval)
        # The non-zero mean adds a systematic bias of -eps_TF32 / 2 * sum to the
        # sum-reduction output, i.e. a constant relative bias of -5e-4. The
        # variance is the same as for round-to-nearest. Per-op behavior:
        #   - max/min: output relative error <= eps_TF32 ~ 1e-3 (one input selected).
        #   - sum of N values: 1-sigma random walk eps_TF32 / (2*sqrt(3)) * sqrt(N) *
        #     |scalar| ~ 0.49 for N=177408, |scalar|_max=4; plus a deterministic
        #     bias eps_TF32 / 2 * |sum| (scales with the output, like rtol).
        #   - mean: same relative behavior as sum; absolute error is sum_error / N.
        #
        # rtol = 5e-3 ~ 5*eps_TF32 absorbs the 5e-4 bias plus 1-sigma random walk
        # for typical |y|. pcc = 0.999 is preserved because TF32 noise has small
        # relative magnitude that does not bias the output *pattern* meaningfully.
        rtol = 5e-3
        pcc = 0.999
        if op in ("sum", "mean") and shape == (3, 4, 8, 56, 33) and dim is None:
            # Scalar output of zero-mean reduction (sum or mean of 177408 randn
            # samples). mean = sum/N preserves relative errors, so both share the
            # same error distribution. Using the model from above, the 1-sigma
            # random walk on the sum is eps_TF32 / (2*sqrt(3)) * sqrt(N) * |scalar|
            # ~ 0.49 (3-sigma ~ 1.46) for N=177408, |scalar|_max=4. The systematic
            # bias is eps_TF32/2 * |sum|, which scales with |sum| and is absorbed
            # by rtol for typical |sum|. bf16 with the same random-walk formula
            # gives 1-sigma ~ 3.89; the user's atol=1.5 covers ~0.4-sigma. Here
            # atol=0.3 covers ~0.6-sigma -- same tuning style (rtol*|y| carries
            # typical |sum|). Mean inherits the default atol since its absolute
            # error is sum_error/N ~ 3e-6.
            #
            # Frobenius on a scalar output degenerates to |error|/|y|, and |y|
            # follows a zero-mean Gaussian (sqrt(N)*|scalar|-stddev for sum,
            # |scalar|/sqrt(N)-stddev for mean) that can land in the lower tail
            # of its distribution and inflate the ratio. 1e-1 covers the
            # practical-unlucky regime where |y| is a small fraction of its stddev.
            atol = 0.3 if op == "sum" else 5e-2
            frobenius_threshold = 1e-1
        elif op == "sum" and shape == (3, 4, 8, 56, 33):
            # Tensor-output sum on the largest shape. Per-element error scales
            # with sqrt(N_reduce); the worst non-None reduction is dim=(0,-2,-1)
            # (N_reduce=5544) with 1-sigma random walk eps_TF32 * sqrt(N_reduce)
            # * |scalar|_max / (2*sqrt(3)) ~ 0.086, so 3-sigma ~ 0.26. Output
            # elements can land near zero by randn cancellation, so atol (not
            # rtol*|y|) must absorb that per-element error: atol = 0.3 covers
            # ~3-sigma. Frobenius is RMS-style on tensor output and stays at
            # ~eps_TF32, so the default threshold applies.
            atol = 0.3
            frobenius_threshold = 5e-3
        else:
            # Smaller shapes (N <= 60), or non-sum ops on the largest shape:
            # max/min absolute error bounded by eps_TF32 * 3 * |scalar|_max ~
            # 0.012; sum on smaller shapes bounded by 1-sigma random walk
            # eps_TF32 * sqrt(N_max) * |scalar|_max / (2*sqrt(3)) ~ 9e-3 for
            # N=60; mean on the largest shape with a non-None dim has
            # per-element error sum_error / N_reduce, smaller still. atol = 5e-2
            # covers all with margin. Per-element relative error of ~eps_TF32
            # ~ 1e-3 gives a relative Frobenius norm of the same order; 5e-3 = 5x margin.
            atol = 5e-2
            frobenius_threshold = 5e-3
    else:
        rtol = 0.05
        if op == "sum" and shape == (3, 4, 8, 56, 33):
            # Summing large number of bfloat16 values accumulates rounding errors,
            # and results also vary from near 0 to relatively large values (in hundreds)
            # PCC should catch any significant errors.
            atol = 1.5
            # dim=None reduces the largest input to a single scalar whose
            # expected value can be near 0 (randn has mean 0), so the relative
            # Frobenius norm ~ atol/|value| can be substantially larger than for
            # multi-element output cases. 0.3 stays well inside what allclose
            # already permits (atol + rtol*|value|) for any |value| >= ~6 and
            # tightens to ~5% for typical |sum| ~ 100.
            frobenius_threshold = 0.3
        else:
            atol = 0.1
            # bf16 default PCC=0.999 implies per-element shape-noise ratio of
            # sqrt(2*(1-PCC)) ~= 4.5%; the relative Frobenius norm is the same
            # order of magnitude on the raw values. 0.1 gives ~2x margin.
            # For var/std on the largest shape, outputs cluster around scalar^2
            # or |scalar| (both >= 1), and per-element absolute error remains
            # bounded by the same ~5% relative bf16 precision, so the Frobenius
            # bound does not need to be loosened the way PCC does.
            frobenius_threshold = 0.1

        if op == "var" and shape == (3, 4, 8, 56, 33):
            # For var/std there are cases where all output values are close to 1, and we're using bfloat16,
            # so even a rounding error of 0.5 ULP has a significant impact on PCC with large tensors.
            pcc = 0.98
        elif op == "std" and shape == (3, 4, 8, 56, 33):
            # For std, sqrtf() adds an extra rounding step on top of variance, further
            # lowering PCC when values cluster near 1.0 (e.g. 3-dim reduction on large tensors).
            # Therefore PCC threshold has to be lower. ATOL/RTOL should catch any significant errors.
            pcc = 0.95
        else:
            pcc = 0.999

    passing, output_msg = assert_numeric_metrics(
        torch_result,
        ttnn_result,
        rtol=rtol,
        atol=atol,
        pcc_threshold=pcc,
        frobenius_threshold=frobenius_threshold,
        assert_on_fail=False,
    )

    assert passing, f"{output_msg}, torch: {torch_result}, ttnn: {ttnn_result}"


# Test that generic reduction ops work correctly with a scalar applied to the input.
# Split into two parts to avoid large test count from full cross product.
# Simple powers of two run with reduced set of dim parameters.
@pytest.mark.parametrize("op, correction", OP_CORRECTION)
@pytest.mark.parametrize("scalar", [1.0, -2.0, 2.0])
@pytest.mark.parametrize("shape, dim", VALID_SHAPE_DIMS_PART1)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_generic_ops_w_scalar_part1(device, op, scalar, correction, dim, shape, dtype):
    _run_generic_ops_w_scalar(device, op, scalar, correction, dim, shape, dtype)


# Non-powers of two and a slightly larger power of two run with full set of dim parameters.
@pytest.mark.parametrize("op, correction", OP_CORRECTION)
@pytest.mark.parametrize("scalar", [-2.43, 2.43, 4.0])
@pytest.mark.parametrize("shape, dim", VALID_SHAPE_DIMS_PART2)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_generic_ops_w_scalar_part2(device, op, scalar, correction, dim, shape, dtype):
    _run_generic_ops_w_scalar(device, op, scalar, correction, dim, shape, dtype)
