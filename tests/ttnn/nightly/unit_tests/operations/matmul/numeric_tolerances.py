# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Numeric tolerances for matmul tests, predicted from the device number formats.

A test compares a matrix multiply run on Tenstorrent hardware against the same
matrix multiply run in PyTorch on the host. The two never agree bit for bit, so
the test has to accept some difference. assert_numeric_metrics, the helper the
tests call to make the comparison, takes four limits:

    atol, rtol (absolute and relative tolerance)
                         every element must satisfy
                         abs(device - reference) <= atol + rtol * abs(reference),
                         so atol is the fixed part of the per element allowance
                         and rtol the part that grows with the element's value
    frobenius_threshold  root-mean-square difference over the whole tensor, as a
                         fraction of the root-mean-square of the reference
    pcc_threshold        smallest Pearson correlation coefficient allowed between
                         the two tensors

matmul_numeric_tolerances returns all four. Tests never call it at run time. You
run it yourself while writing a test and paste the numbers in, so the limits stay
plain constants where they are used, and so editing this module cannot quietly
move what every test accepts.

The matmul can also apply an activation function to its result on device. The
test still compares against the reference with the activation applied (below, the
activated reference), but what you hand this module is the product before the
activation, because working out how the activation transforms an error needs the
values going into it. You pass the activation itself twice: as the device side op
(from ttnn, the Python interface to the device ops), which the module uses to
look up how accurately the hardware evaluates that activation, and as the
equivalent PyTorch function, which the module evaluates to find how much the
activation stretches or compresses an error. Accuracy varies by activation: the
device evaluates some as a fitted polynomial and others from a piecewise linear
lookup table, which is much coarser.

The calculation runs in two stages.

Stage one: a single relative error. matmul_relative_error predicts one relative
error r from the configuration only, never from the data. r is defined so that
across the whole output tensor the predicted error has a standard deviation of r
times the standard deviation of the reference. It reads these inputs:

  - how many mantissa bits each number format keeps, from the inputs through to
    the partial sums. The final rounding into the output format is not part of r;
    stage two adds it, because it scales with the element's own value
  - K, the reduction length, which is the shared dimension of the two operands
  - math_fidelity, which selects how many passes the multiply hardware makes and
    so how much of each input's mantissa it uses
  - k_tiles_per_block, how many 32 element slices of K are summed before the
    partial sum is written out and rounded. The hardware works in 32 by 32 tiles,
    and fewer slices per write means more roundings along K
  - fp32_dest_acc_en, whether partial sums are held in 32 bit rather than 16 bit,
    and packer_l1_acc, whether a new partial sum is added straight into the
    output buffer as it is written, instead of the value there being read back,
    so that one rounding to the output format is skipped

Every rounding step in that chain contributes one relative error term. K and
k_tiles_per_block set how many times each step happens; math_fidelity and the two
accumulation settings set how large each term is. The terms are combined as the
square root of the sum of their squares, the usual rule for independent errors.

Stage two: the four limits. Turning r into them needs the reference tensor as
well, because r is only a ratio: atol is an absolute number, and the activation
corrections depend on the reference values.

- atol. Each output element is a sum of K products. The model takes the error of
  that sum to be about the same size for every element, because it comes from
  rounding partial sums, whose typical magnitude is the same across the output,
  not from the size of the element it lands in. Stage one already fixed that
  size, r times the standard deviation of the reference, and taking it to be the
  same everywhere is what lets one number serve as atol for the whole tensor.
  That error is itself a sum of many independent roundings, so it is close to
  Gaussian. Write n for the number of output elements. An error can fall either
  way, so the largest absolute error among n elements is about as extreme as the
  largest of 2n one sided samples, which for a Gaussian sits about
  sqrt(2 * ln(2n)) standard deviations from zero. The module then shifts each
  reference value by that distance, up and down, applies the activation to all
  three values, and keeps the largest change in the activation's output over
  every element and both directions. It does not multiply the distance by the
  activation's slope: at worst case distance the activation is nowhere near
  straight. This is why an activation that flattens out at both ends, such as
  sigmoid, is never assigned more error than its whole output range. The device's
  own absolute error in evaluating the activation is then added.

- rtol takes the two contributions that do scale with an element's own value: the
  activation's relative error and the rounding into the output format. r does not
  enter it. The two are added rather than combined in quadrature, so that rtol
  stays a bound for every element rather than a typical value. For an activation
  the device evaluates as a polynomial the activation's own error dominates, 87
  percent of the total in the example below.

- frobenius_threshold combines three terms as the square root of the sum of their
  squares: r times the standard deviation of the reference, since that is how r
  is defined, multiplied by the root-mean-square of the activation's slope taken
  over the reference values; the sum of the output format rounding and the
  activation's relative error, multiplied by the root-mean-square of the
  activated reference; and the device's absolute error on the activation. The
  result is divided by the root-mean-square of the activated reference. A slope
  is enough here, where atol needed an evaluation, because this term describes a
  typical error, far smaller than the worst case atol has to cover, and the
  activation is nearly linear over an interval that small.

- pcc_threshold reuses that combined error, before it is divided. Write q for the
  combined error divided by the standard deviation of the activated reference,
  the standard deviation and not the root-mean-square because a correlation
  removes the mean of each tensor first. An error independent of the reference
  values adds to the device tensor's variance without adding to the covariance,
  which scales the correlation by 1 / sqrt(1 + q * q). matmul_numeric_tolerances
  returns the small q approximation of that, 1 - q * q / 2.

atol, rtol and frobenius_threshold are multiplied by the safety argument, 2.0 by
default. pcc_threshold does not use it: q is always multiplied by sqrt(1.5), a
fixed module constant. Because a correlation's distance from 1 goes as the square
of the error, that is a margin of 1.5 on the distance from 1.

To get limits for a new entry in a test's pytest.mark.parametrize list, build the
reference tensor the test will compare against and call matmul_numeric_tolerances
with it. Run this from a Python session whose working directory is the repository
root, so that the import resolves:

    import torch
    import ttnn
    import torch.nn.functional as F
    from tests.ttnn.nightly.unit_tests.operations.matmul.numeric_tolerances import (
        matmul_numeric_tolerances,
    )

    M, K, N = 128, 512, 512
    torch.manual_seed(0)
    # bfloat16 is a 16 bit float that stores 7 mantissa bits. Rounding the
    # inputs to it here is what the device does when it receives them.
    in0 = torch.randn(1, 1, M, K).bfloat16().float()
    in1 = torch.randn(1, 1, K, N).bfloat16().float()

    print(
        matmul_numeric_tolerances(
            in0 @ in1,                                    # reference, pre activation
            ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU),   # as the device applies it
            F.gelu,                                       # the PyTorch equivalent
            K=K,
            in0_dtype=ttnn.bfloat16,
            in1_dtype=ttnn.bfloat16,
            out_dtype=ttnn.bfloat16,
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
            k_tiles_per_block=2,
        )
    )

which prints

    {'atol': 6.565339088439941, 'rtol': 0.035760548978043954,
     'frobenius_threshold': 0.06724930087282528,
     'pcc_threshold': 0.9987590077261392}

The two per element limits can be followed by hand. For atol, r is 0.0284 for
this configuration and the reference has a standard deviation of 22.6, so one
element's error has a standard deviation of 0.64. The output has M * N =
128 * 512 = 65536 elements, so 2n is 131072 and sqrt(2 * ln(131072)) is 4.855,
giving a worst element error of 4.855 * 0.64 = 3.11. Shifting the reference by
that distance and reapplying GELU raises it to 3.28: the largest change is at a
reference value of 0.75, where GELU gives 0.58, while at 3.86, which is 3.11
higher, GELU is close enough to the identity to give 3.87, a change of 3.28. The
module records no absolute error for GELU, because the device fits it with a
polynomial and that inaccuracy is carried as the relative error in rtol instead.
The safety argument of 2.0 gives the printed 6.57.

For rtol, a unit in the last place is the gap between neighbouring representable
values, which for a 7 bit mantissa is 2**-7 of the value. The module's table
gives GELU a relative error of two of those in whatever format the accumulator
holds, which is bfloat16 here because fp32_dest_acc_en is False, so 2 * 2**-7 =
0.0156. Rounding into a bfloat16 output moves a value by at most half a unit in
the last place, 2**-8; taking that error as uniform over the rounding interval
gives a root-mean-square relative error of 2**-8 / sqrt(3) = 0.0023. Twice their
sum is the printed 0.0358. The two tensor wide limits use the same pieces through
the formulas above and are not traced here.

One parametrize entry usually covers several shapes, data types and settings. Run
the calculation for each and keep the largest atol, rtol and
frobenius_threshold and the smallest pcc_threshold. Round the first three up and
pcc_threshold down.

Notes:

- The relative error of a matmul does not grow with K. Each product carries a
relative rounding error from the narrowed mantissas. Because inputs to the tests
include both positive and negative random values (from torch.randn), those errors
multiply values of random sign, so the total error grows as sqrt(K), and so
does the result itself. As far as relative error is concerned, they cancel out.
The terms that do grow are the two that round a running total: the accumulator
itself, once per 16 elements of K per fidelity pass, and the partial sum spilled
between K blocks. Both grow as sqrt(K). Both scale with the accumulator
format, so with fp32_dest_acc_en they fall below every other term and
K stops mattering at all; they are only significant with a 16 bit accumulator.

- Truncation is biased: it always errs in the same direction. The matrix engine
drops the low mantissa bits of each operand instead of rounding to nearest, which
moves that operand toward zero, so no product comes out larger in magnitude than
the true result. Consider the product of a and b as a*b*(1-d), where d is the
fraction of the true product lost from the two truncated operands together.
d is always positive because the result is always truncated toward zero.
One might expect a loss that never changes sign to add up over the K products,
making the total grow linearly with K.
It does not, because only d has a fixed sign. The products a*b can be either
positive or negative, so the errors -d*a*b can be either positive or negative
and cancel in the sum just as the products themselves do. Therefore, relative
error does not grow with K.

The bias does change how the truncation term enters r. A source whose error
averages to zero contributes the standard deviation of its error divided by the
standard deviation of the result. Truncation does not average to zero. Split
the error one product contributes into a bias part, -mu*a*b where mu is the
average of d, and a noise part, -(d-mu)*a*b: the bias part sums to exactly the
true result shrunk by mu, and only the noise part averages to zero. The
root-mean-square of d is the square root of the sum of mu squared and the
variance, so using it in place of a standard deviation carries the bias into
the same sum of squares as every other source. That is not exact for a fixed
offset, it is a simple upper bound, and _truncation_relative_error, which
computes this root-mean-square for one operand, measures the margin it leaves.
"""

import math

import torch
import ttnn

_SQRT3 = math.sqrt(3.0)
_SQRT6 = math.sqrt(6.0)

# Margin on the correlation limit, as a multiplier on the relative error. Squared
# by the conversion to a correlation, so it allows 1.5x on the distance from 1.
# The whole-tensor and elementwise limits take the larger safety factor
# instead; a correlation needs less because the quantity it is built from is a
# root-mean-square over tens of thousands of elements, which barely varies from
# run to run. Some margin is still needed: with the truncation term corrected the
# budget tracks the measured error closely rather than sitting well above it.
_PCC_MARGIN = math.sqrt(1.5)

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

# Mantissa bits the matrix engine keeps from each operand, as (SrcA, SrcB), taken
# from the union of the per-pass bit masks. The multiplier is 5 bits by 7 bits
# and each fidelity level makes another pass to consume more of the inputs
# (tech_reports/matrix_engine/matrix_engine.md). A matmul loads the right hand
# operand into SrcA and the left hand operand into SrcB, the opposite of other
# operations, so the right hand operand takes the narrower path.
#
# Over an 11 bit significand field with the hidden bit at position 10, the masks
# cover SrcA 10..6 then 5..1, and SrcB 10..4 then 3..0. Accumulating those:
#   LoFi   pass 0        SrcA 4 bits,  SrcB 6 bits
#   HiFi2  passes 0,1    SrcA 9 bits,  SrcB 6 bits
#   HiFi3  passes 0,1,2  SrcA 9 bits,  SrcB 10 bits
#   HiFi4  all passes    same coverage as HiFi3, plus the low-by-low cross term
# Nine and ten bits exceed what any input format here carries, so those operands
# are consumed whole and contribute nothing.
_FIDELITY_MANTISSA_BITS_KEPT = {
    ttnn.MathFidelity.LoFi: (4, 6),
    ttnn.MathFidelity.HiFi2: (9, 6),
    ttnn.MathFidelity.HiFi3: (9, 10),
    ttnn.MathFidelity.HiFi4: (9, 10),
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
    """Contribution to r from one rounding of a value into dtype.

    For a normalised format this is the rounding error relative to the value.
    For a block float format the step is fixed across a block of 16 by the
    largest magnitude in it, which for normally distributed data is about two
    standard deviations, so the error is absolute rather than relative. An
    absolute error of k standard deviations on an operand contributes k
    to r in exactly the same way a relative one does, which is why both
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
    """Contribution to r from dropping mantissa bits, for one operand.

    The operand arrives with bits_available mantissa bits and the matrix engine
    keeps only bits_kept of them. If it keeps at least as many as arrive, nothing
    is dropped and the contribution is zero.

    The mantissa bits encode a significand, a value at least 1 and below 2.
    Dropping the bits below bits_kept removes less than the value of the lowest
    kept bit, one unit in the last kept place, u = 2**-bits_kept. Taking the
    amount removed as equally likely anywhere from zero to u, its average is u/2
    and its standard deviation is u/sqrt(12), the standard deviation of a uniform
    range of width u. Its root-mean-square is the square root of the sum of the
    average squared and the standard deviation squared, u*sqrt(1/4 + 1/12), which
    is u/sqrt(3). Truncation is biased, so the root-mean-square is the figure to
    use, as the module docstring explains; the standard deviation alone would be
    a factor of two lower.

    What r needs is the loss relative to the operand's own value. The amount
    removed and the value share the same power of two, so that factor cancels and
    the relative loss is the amount removed divided by the significand alone.
    Taking the amount removed as independent of the significand, the mean square
    of the ratio is the mean square of the amount removed times the average of
    one over the square of the significand. Taking the significand as equally
    likely anywhere in its range, that average is exactly a half: the area under
    1/s**2 from s = 1 to s = 2 is 1 - 1/2, and the range has width 1, so that
    area is the average. The relative root-mean-square is therefore u/sqrt(3)
    times sqrt(1/2), which is u/sqrt(6). The same division by the significand
    turns the average u/2 into the mu of the module docstring.

    Both uniform assumptions are approximations, so the formula is checked
    against measurement. Truncating bfloat16 values drawn from torch.randn, the
    distribution the tests use, which carry 7 mantissa bits, and measuring the
    root-mean-square relative loss on one operand gives 0.0236, 0.0107 and 0.0041
    for 4, 5 and 6 kept bits, that is 3, 2 and 1 dropped bits, against this
    formula's 0.0255, 0.0128 and 0.0064. The formula is the larger number every
    time, by 8, 20 and 57 percent. The overstatement grows as fewer bits are
    dropped, because the amount removed then takes only a few discrete values
    rather than a continuous spread: its root-mean-square is 0.523u with three
    dropped bits and 0.354u with one, where the continuous figure of
    u/sqrt(3) = 0.577u is 1.10 and 1.63 times those.

    An understatement in how the caller combines the operands nearly cancels that
    overstatement. A product loses from both operands, and matmul_relative_error
    adds the two contributions as the square root of the sum of squares. Squaring
    the sum of the two relative losses produces a term in the two biases together
    that a sum of squares leaves out, so the correct figure for the pair is 1.16
    times what the caller computes. For bfloat16 operands at LoFi, the engine's
    lowest precision setting, which keeps 4 mantissa bits of one operand and 6 of
    the other, the two effects very nearly cancel and the caller's figure lands
    0.5 percent above the measured pair. This term therefore carries almost no
    margin of its own and relies on the safety factor applied to the limits.
    """
    if bits_kept >= bits_available:
        return 0.0
    return 2.0**-bits_kept / _SQRT6


def _accumulated_rounding_relative_error(step_error, steps):
    """Contribution to r from rounding a running total steps times.

    These roundings act on the partial sum rather than on individual products,
    so they do not cancel against the sqrt(K) growth of the result. After
    j of n steps the partial sum is sqrt(j / n) of the final size, and
    the error introduced there survives to the end, so the variances sum to
    step_error**2 * (n + 1) / 2 times the square of the final size.
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
            only the final value, because without packer_l1_acc the partial
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

    The truncation sources carry a bias and not only noise, and that reaches the
    limits built from this number unevenly. The elementwise absolute limit and
    the whole-tensor relative error limit both need it counted, since it moves
    every element toward zero. The minimum correlation does not: scaling every
    element by close to the same factor barely changes the correlation with the
    reference, so counting the bias asks for less correlation than would be
    justified, which cannot cause a false failure but does let more real error
    through.
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
    """Error of evaluating activation on device, as (relative, absolute).

    dest_dtype is the format the activation runs in, which is the
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
    """Tolerances for assert_numeric_metrics on a matmul with a fused activation.

    Splat the result into the assertion:

        assert_numeric_metrics(golden, actual, check_ulp=False, **tolerances)

    Args:
        pre_activation: the exact host reference for the matmul, and the bias if
            there is one, before the activation. Its spread sets the scale of the
            predicted error, so it is needed even when the reference the test
            finally compares against is the activated one.
        activation: the ttnn.UnaryWithParam (or None) the device applied,
            used to work out how accurately the device evaluates it.
        activation_fn: the same activation as a callable on a torch tensor, used
            to carry the predicted error through it. None means no activation.
        safety: multiplier on the predicted error, applied to atol, rtol
            and frobenius_threshold. Those scale linearly with the error, so
            a factor of 2 stays a factor of 2. Since independent errors combine
            in quadrature it covers a missed term up to sqrt(3) times the
            largest one already counted, and for the elementwise limit it also
            covers the run-to-run spread of an extreme value.

            It is not applied to pcc_threshold, which takes the smaller
            _PCC_MARGIN instead. The distance of a correlation from 1 goes
            as the square of the relative error, so this factor of 2 would
            become a factor of 4 there. For an activation that saturates, where
            the reference's variance collapses and the distance from 1 is large
            enough to see, that produces a limit several times looser than the
            hardware needs.

    Other arguments are as for matmul_relative_error.

    Returns:
        A dict with atol, rtol, frobenius_threshold and pcc_threshold.
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
    # rounding error of K products. A bias is added after the matmul and carries
    # none of that error; it is left in here because the tests that use one make
    # it far smaller than the matmul result.
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
    # _PCC_MARGIN rather than safety: see the note on safety above.
    if golden_std > 0.0:
        noise_ratio = _PCC_MARGIN * rms_error / golden_std
        tolerances["pcc_threshold"] = max(0.0, min(0.999999, 1.0 - noise_ratio * noise_ratio / 2.0))
    else:
        tolerances["pcc_threshold"] = 0.0

    return tolerances
