# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Numeric tolerances for matmul tests, predicted from the device number formats.

Nothing imports this module. The tests hold their limits as literals so that the
limit being applied is visible at the point it is applied, and so that editing the
model here cannot quietly move what every test accepts. This module is the record
of where those literals came from: run its functions by hand to re-derive a table
when a test's shapes, seeds, data types or activations change, and paste the
result back. A literal is set to the most permissive value the model gives over
the cases it covers, rounded outward, so that no case is held to a tighter limit
than the model asks for.

A matmul on device and the same matmul in PyTorch never agree bit for bit, so a
test has to accept some difference. Choosing that limit by raising it until the
test passes produces limits that no longer detect a wrong answer: a limit of
``rtol = 10 * K`` grants every element an allowance thousands of times its own
value, and a relative Frobenius limit above 1.0 is passed by a tensor of zeros.

The functions here predict the error instead, from the width of each number
format on the path and the number of times a value is rounded into one. The
prediction is then the limit. Nothing here measures the device.

The error is summarised as a single **relative error** ``r``, meaning the
predicted error on a result has standard deviation ``r`` times the standard
deviation of that result. Independent sources combine as ``sqrt(sum of
squares)``.

Two facts shape everything below.

*The relative error of a matmul does not grow with K.* Each product carries a
fractional slip from the narrowed mantissas. Those slips multiply values of
random sign, so the total error grows as ``sqrt(K)``, and so does the result
itself. They cancel. The one term that does grow is accumulator rounding, and it
grows as ``sqrt(K)``, never as ``K``.

*A one-sided slip behaves like a random one.* Mantissa bits are dropped rather
than rounded, so every slip has the same sign. It still does not build up,
because it multiplies a mean-zero quantity. The figure that matters is therefore
the root-mean-square of the slip including its mean, ``2**-m / sqrt(3)`` for a
slip spread over ``[0, 2**-m)``, rather than its standard deviation.
"""

import math

import torch
import ttnn

_SQRT3 = math.sqrt(3.0)

# Stored mantissa bits, excluding the leading one bit that normalised floating
# point formats imply rather than store. The gap between neighbouring values is
# 2**-bits relative to that leading bit.
_MANTISSA_BITS = {
    ttnn.float32: 23,
    ttnn.bfloat16: 7,
}

# Block float formats share one exponent between 16 consecutive values and store
# the leading one bit explicitly, so the usable magnitude bits are one fewer than
# the field width suggests: bfloat8_b spends 8 bits as 1 sign + 7 magnitude and
# has 6 bits of real precision (tech_reports/data_formats/data_formats.md).
_BLOCK_FLOAT_BITS = {
    ttnn.bfloat8_b: 6,
    ttnn.bfloat4_b: 2,
}

# Mantissa bits the matrix unit keeps from each operand, as (SrcA, SrcB).
# The multiplier is 5 bits by 7 bits and each fidelity level makes another pass
# over the inputs to consume more of them
# (tech_reports/matrix_engine/matrix_engine.md). A matmul loads the right hand
# operand into SrcA and the left hand operand into SrcB, the opposite of other
# operations, so the right hand operand takes the narrower path.
_FIDELITY_MANTISSA_BITS_KEPT = {
    ttnn.MathFidelity.LoFi: (4, 6),
    ttnn.MathFidelity.HiFi2: (7, 6),
    ttnn.MathFidelity.HiFi3: (4, 7),
    ttnn.MathFidelity.HiFi4: (7, 7),
}

# Passes over the inputs, which is also how many times each fidelity level
# re-runs the multiply sequence and accumulates on top of the same result.
_FIDELITY_PASSES = {
    ttnn.MathFidelity.LoFi: 1,
    ttnn.MathFidelity.HiFi2: 2,
    ttnn.MathFidelity.HiFi3: 3,
    ttnn.MathFidelity.HiFi4: 4,
}

# One matmul instruction reduces 16 elements of K, and two of them contribute to
# each output element per 32 element K tile, so the running total is rounded into
# the accumulator once per 16 elements of K per fidelity pass.
_K_PER_ACCUMULATION = 16


def _format_relative_error(dtype):
    """Contribution to ``r`` from one rounding of a value into ``dtype``.

    For a normalised format this is the rounding error relative to the value.
    For a block float format the step is fixed across a block of 16 by the
    largest magnitude in it, which for normally distributed data is about two
    standard deviations, so the error is absolute rather than relative. An
    absolute error of ``k`` standard deviations on an operand contributes ``k``
    to ``r`` in exactly the same way a relative one does, which is why both
    return a single comparable number.
    """
    bits = _BLOCK_FLOAT_BITS.get(dtype)
    if bits is not None:
        # Step at most 2 * 2**-bits standard deviations, rounded to nearest.
        return 2.0**-bits / _SQRT3
    return 2.0 ** -(_MANTISSA_BITS[dtype] + 1) / _SQRT3


def _source_mantissa_bits(dtype):
    """Mantissa bits the data actually carries when it reaches the multiplier."""
    bits = _BLOCK_FLOAT_BITS.get(dtype)
    return bits if bits is not None else _MANTISSA_BITS[dtype]


def _truncation_relative_error(bits_kept, bits_available):
    """Contribution to ``r`` from dropping mantissa bits below ``bits_kept``."""
    if bits_kept >= bits_available:
        return 0.0
    return 2.0**-bits_kept / _SQRT3


def _accumulated_rounding_relative_error(step_error, steps):
    """Contribution to ``r`` from rounding a running total ``steps`` times.

    These roundings act on the partial sum rather than on individual products,
    so they do not cancel against the ``sqrt(K)`` growth of the result. After
    ``j`` of ``n`` steps the partial sum is ``sqrt(j / n)`` of the final size, and
    the error introduced there survives to the end, so the variances sum to
    ``step_error**2 * (n + 1) / 2`` times the square of the final size.
    """
    if steps < 1:
        return 0.0
    return step_error * math.sqrt((steps + 1) / 2.0)


def matmul_relative_error(
    K,
    in0_dtype,
    in1_dtype,
    out_dtype,
    math_fidelity,
    fp32_dest_acc_en,
    packer_l1_acc,
    k_tiles_per_block=1,
    reference_dtype=None,
):
    """Predicted relative error of a matmul result, before any fused activation.

    Args:
        K: the reduction length.
        in0_dtype, in1_dtype: device formats of the left and right operands.
        out_dtype: device format of the result. It reaches the accumulation, not
            only the final value, because without ``packer_l1_acc`` the partial
            sums are spilled to memory in this format.
        math_fidelity: how many passes the matrix unit makes over the inputs.
        fp32_dest_acc_en: whether the accumulator is 32 bit rather than 16 bit.
        packer_l1_acc: whether partial sums are accumulated in place in memory
            instead of being written out and read back once per K block.
        k_tiles_per_block: K tiles per block, which sets how many blocks the
            reduction is split into. The default of 1 gives the most blocks and
            so the most partial sum roundings.
        reference_dtype: format the host reference was rounded to, if it was not
            computed in float32.

    Returns:
        The relative error as a float.
    """
    srca_bits, srcb_bits = _FIDELITY_MANTISSA_BITS_KEPT[math_fidelity]
    terms = [
        # Converting each operand to its device format.
        _format_relative_error(in0_dtype),
        _format_relative_error(in1_dtype),
        # The multiplier discarding the low mantissa bits of each operand.
        _truncation_relative_error(srcb_bits, _source_mantissa_bits(in0_dtype)),
        _truncation_relative_error(srca_bits, _source_mantissa_bits(in1_dtype)),
    ]

    accumulator_dtype = ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16
    accumulations = (K // _K_PER_ACCUMULATION) * _FIDELITY_PASSES[math_fidelity]
    terms.append(_accumulated_rounding_relative_error(_format_relative_error(accumulator_dtype), accumulations))

    blocks = max(1, (K // ttnn.TILE_SIZE) // max(1, k_tiles_per_block))
    if blocks > 1:
        if fp32_dest_acc_en:
            partial_dtype = ttnn.float32
        elif packer_l1_acc:
            partial_dtype = ttnn.bfloat16
        else:
            partial_dtype = out_dtype
        terms.append(_accumulated_rounding_relative_error(_format_relative_error(partial_dtype), blocks - 1))

    if reference_dtype is not None:
        terms.append(_format_relative_error(reference_dtype))

    return math.sqrt(sum(term * term for term in terms))


def _op_type(activation):
    return getattr(activation, "op_type", None)


# Activations the fused matmul path evaluates with comparison and selection only,
# so they carry no error beyond the rounding of the value they produce. relu is
# not even a vector unit operation there: it is a packer setting.
_EXACT_ACTIVATIONS = frozenset(
    {
        ttnn.UnaryOpType.RELU,
        ttnn.UnaryOpType.RELU6,
        ttnn.UnaryOpType.HARDTANH,
    }
)

# Every other activation the matmul can fuse is a fitted polynomial or an
# exponential. The implementations record maximum errors between 0.28 and 1 unit
# in the last place of the format they run in; two units is the group bound.
_FITTED_ACTIVATION_ULPS = 2.0

# gelu, tanh and sigmoid each accept a flag that replaces the fit with a handful
# of straight line segments. The bound is the peak distance from those segments
# to the true function, so it is absolute and does not shrink with the output.
#
# tanh's three segments are written out in the kernel that loads them: 0.90625*x,
# then 0.09375*x + 0.8125, then 1. They join at x = 1 and x = 2, and their
# greatest distance from tanh is at x = 1, where 0.90625 - tanh(1) = 0.145.
#
# The gelu and sigmoid tables are loaded as packed constants with no arithmetic
# form recorded beside them, so their bounds are not derived here. 0.13 is the
# figure the vector unit's own sweep tests record for both
# (tt_metal/tt-llk/tests/python_tests/test_eltwise_unary_sfpu.py), noting that
# the absolute error of a coarse table peaks near the segment joins. This is the
# one quantity in this module taken from an observation rather than from a
# format width or a written algorithm.
_PIECEWISE_LINEAR_ABSOLUTE_ERROR = {
    ttnn.UnaryOpType.GELU: 0.13,
    ttnn.UnaryOpType.TANH: 0.15,
    ttnn.UnaryOpType.SIGMOID: 0.13,
}

# Index of the parameter that selects the piecewise linear table. For sigmoid the
# first parameter is the vector mode and the flag is the second one.
_PIECEWISE_LINEAR_FLAG_INDEX = {
    ttnn.UnaryOpType.GELU: 0,
    ttnn.UnaryOpType.TANH: 0,
    ttnn.UnaryOpType.SIGMOID: 1,
}

# The root-mean-square error of a piecewise linear fit is well below its peak,
# because the segments meet the curve at their ends. Half the peak is used for
# the whole-tensor checks, while the peak itself is used elementwise.
_PIECEWISE_LINEAR_RMS_FRACTION = 0.5

# Most activations pick a higher accuracy inner algorithm when the accumulator is
# 32 bit. These two do not: they select it from a compile time macro that only the
# eltwise unary path defines, so fused into a matmul they always take their 16 bit
# polynomial branch however the accumulator is configured.
_ALWAYS_16_BIT_ACTIVATIONS = frozenset(
    {
        ttnn.UnaryOpType.SELU,
        ttnn.UnaryOpType.SOFTPLUS,
    }
)


def activation_evaluation_error(activation, dest_dtype):
    """Error of evaluating ``activation`` on device, as (relative, absolute).

    ``dest_dtype`` is the format the activation runs in, which is the
    accumulator format rather than the output format: the activation is applied
    to the value still held in the accumulator, before it is packed out.
    """
    op_type = _op_type(activation)
    if op_type is None or op_type in _EXACT_ACTIVATIONS:
        return 0.0, 0.0

    flag_index = _PIECEWISE_LINEAR_FLAG_INDEX.get(op_type)
    if flag_index is not None:
        params = activation.params
        if len(params) > flag_index and params[flag_index] != 0.0:
            return 0.0, _PIECEWISE_LINEAR_ABSOLUTE_ERROR[op_type]

    if op_type in _ALWAYS_16_BIT_ACTIVATIONS:
        dest_dtype = ttnn.bfloat16

    return _FITTED_ACTIVATION_ULPS * 2.0 ** -_MANTISSA_BITS[dest_dtype], 0.0


def matmul_numeric_tolerances(
    pre_activation,
    activation=None,
    activation_fn=None,
    *,
    K,
    in0_dtype,
    in1_dtype,
    out_dtype,
    math_fidelity,
    fp32_dest_acc_en,
    packer_l1_acc,
    k_tiles_per_block=1,
    reference_dtype=None,
    safety=2.0,
):
    """Tolerances for ``assert_numeric_metrics`` on a matmul with a fused activation.

    Splat the result into the assertion::

        assert_numeric_metrics(golden, actual, check_ulp=False, **tolerances)

    Args:
        pre_activation: the exact host reference for the matmul, and the bias if
            there is one, before the activation. Its spread sets the scale of the
            predicted error, so it is needed even when the reference the test
            finally compares against is the activated one.
        activation: the ``ttnn.UnaryWithParam`` (or ``None``) the device applied,
            used to work out how accurately the device evaluates it.
        activation_fn: the same activation as a callable on a torch tensor, used
            to carry the predicted error through it. ``None`` means no activation.
        safety: multiplier on the predicted error. For the whole-tensor checks
            the prediction is a root-mean-square over tens of thousands of
            elements, so it barely varies from run to run and the margin covers
            the model missing a term. Since independent errors combine in
            quadrature, a factor of 2 covers a missed term up to sqrt(3) times
            the largest one already counted. The elementwise limit is different:
            it rests on an extreme value, which does vary noticeably from run to
            run, so there the margin covers both.

    Other arguments are as for :func:`matmul_relative_error`.

    Returns:
        A dict with ``atol``, ``rtol``, ``frobenius_threshold`` and
        ``pcc_threshold``.
    """
    pre_activation = pre_activation.float()

    relative_error = matmul_relative_error(
        K=K,
        in0_dtype=in0_dtype,
        in1_dtype=in1_dtype,
        out_dtype=out_dtype,
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=packer_l1_acc,
        k_tiles_per_block=k_tiles_per_block,
        reference_dtype=reference_dtype,
    )

    # The error on an output element is proportional to the spread of the whole
    # dot product, not to that element's own value, because it is the accumulated
    # slip of K products. A bias is added after the matmul and carries none of
    # that error; it is left in here because the tests that use one make it far
    # smaller than the matmul result.
    error_before_activation = relative_error * pre_activation.std().item()

    # Largest error a single element can plausibly receive before the activation.
    # The errors are a sum of many independent contributions and so are close to
    # normally distributed. The elementwise check bounds the magnitude of the
    # error, so the statistic wanted is the largest of n absolute values, which
    # sits at about sqrt(2 * ln(2n)) standard deviations rather than
    # sqrt(2 * ln(n)).
    extreme_value_factor = math.sqrt(2.0 * math.log(2 * max(pre_activation.numel(), 2)))
    extreme_error_before_activation = extreme_value_factor * error_before_activation

    if activation_fn is None:
        golden = pre_activation
        slope_rms = 1.0
        max_propagated_error = extreme_error_before_activation
    else:
        golden = activation_fn(pre_activation).float()
        # A symmetric difference taken at the scale of the error, rather than at
        # a vanishing scale, gives the average slope over the interval the error
        # can move the value. A corner such as relu's kink or hardtanh's clamp
        # then contributes a partial slope instead of an undefined one.
        step = error_before_activation if error_before_activation > 0.0 else 1e-6
        slope = ((activation_fn(pre_activation + step) - activation_fn(pre_activation - step)) / (2.0 * step)).float()
        slope_rms = slope.pow(2).mean().sqrt().item()
        # For the elementwise bound the activation is evaluated at the extreme of
        # the error rather than multiplied by a slope, so that a bounded
        # activation cannot be credited with more error than its range allows.
        shift = extreme_error_before_activation if extreme_error_before_activation > 0.0 else 1e-6
        moved_up = (activation_fn(pre_activation + shift).float() - golden).abs()
        moved_down = (activation_fn(pre_activation - shift).float() - golden).abs()
        max_propagated_error = torch.maximum(moved_up, moved_down).max().item()

    accumulator_dtype = ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16
    activation_relative, activation_absolute = activation_evaluation_error(activation, accumulator_dtype)
    output_relative = _format_relative_error(out_dtype)

    golden_rms = golden.pow(2).mean().sqrt().item()
    golden_std = golden.std().item()

    # Whole-tensor error, as a root-mean-square over elements.
    rms_error = math.sqrt(
        (slope_rms * error_before_activation) ** 2
        + ((activation_relative + output_relative) * golden_rms) ** 2
        + (_PIECEWISE_LINEAR_RMS_FRACTION * activation_absolute) ** 2
    )

    # Largest error over all elements. Every element is credited with the extreme
    # error rather than only the unluckiest one, so this is an upper bound rather
    # than an estimate. The activation's own error is already a bound, so it is
    # added rather than scaled.
    max_error = max_propagated_error + activation_absolute

    # rtol carries only what genuinely scales with an element's own value: the
    # rounding into the output format and the activation's relative error. atol
    # carries the accumulated error, which does not.
    tolerances = {
        "atol": safety * max_error,
        "rtol": safety * (output_relative + activation_relative),
        "frobenius_threshold": safety * rms_error / golden_rms if golden_rms > 0.0 else safety * rms_error,
    }

    # Writing the device result as the reference plus independent noise of
    # relative size r gives a correlation of 1 / sqrt(1 + r**2), or about
    # 1 - r**2 / 2. The correlation is taken after each tensor has its mean
    # removed, so the divisor is the reference's standard deviation and not its
    # root-mean-square; for a saturating activation such as sigmoid those differ
    # by a lot, because most of the reference's size is in its mean.
    if golden_std > 0.0:
        noise_ratio = safety * rms_error / golden_std
        tolerances["pcc_threshold"] = max(0.0, min(0.999999, 1.0 - noise_ratio * noise_ratio / 2.0))
    else:
        tolerances["pcc_threshold"] = 0.0

    return tolerances
