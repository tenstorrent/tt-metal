# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Matmul test tolerances, derived from the mantissa bits lost at each step.

Matmul tests compare a matrix multiply run on Tenstorrent hardware against the
same matrix multiply run in PyTorch on the host. The two never agree bit for bit,
so the test has to accept some difference. assert_numeric_metrics, the helper the
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

matmul_numeric_tolerances returns all four. You run it yourself while writing a
test and paste the numbers into the test, so the limits stay plain constants where
they are used, and so editing this module cannot quietly move what every test
accepts. The predictions assume test inputs drawn from a normal distribution, so
a mix of positive and negative numbers, which is what the tests generate.

The matmul can also apply an activation function to its result on device (a fused
activation). The test compares against the activated reference, the product with
the activation applied, but what you hand this module is the pre-activation
reference, the product before it. Working out how the activation transforms an
error needs the values going into it, and the activated reference cannot supply
them: an activation that saturates (i.e. it flattens out at both ends), is not
invertible.

You pass the activation twice: as the device side op (from ttnn, the Python
interface to the device ops), which fixes how accurately the hardware evaluates
it, and as the equivalent PyTorch function, which the module evaluates to find
how much the activation stretches or compresses an error. Nothing checks that the
two agree, so pass the function that matches the op.

How accurately the hardware evaluates an activation varies. relu, relu6 and
hardtanh are exact, being comparison and selection only. The rest are fitted
polynomials, held to two units in the last place. gelu, tanh and sigmoid also
accept a flag that swaps the fit for a handful of straight line segments, which
is far coarser; see _PIECEWISE_LINEAR_ABSOLUTE_ERROR for how loose those cases
end up, and why gelu's figure is the one number in this module that is measured
rather than derived.

Stage one: a single relative error. matmul_relative_error predicts one relative
error "r" from the configuration only, never from the data. "r" is defined so that
across the whole output tensor the predicted error has a standard deviation of "r"
times the standard deviation of the pre-activation reference. It reads these
inputs:

  - how many mantissa bits each number format keeps, from the inputs through to
    the partial sums. The final rounding into the output format is not part of "r";
    stage two adds it, because it scales with the element's own value
  - K, the reduction length, which is the shared dimension of the two operands
  - math_fidelity, which selects how many passes the matrix engine makes and so
    how much of each input's mantissa it uses
  - k_tiles_per_block, how many 32 element slices of K are summed before the
    partial sum is written out and rounded. The hardware works in 32 by 32 tiles,
    and fewer slices per write means more roundings along K
  - fp32_dest_acc_en, whether partial sums are held in 32 bit rather than 16 bit,
    and packer_l1_acc, whether the packer adds each new partial sum into the buffer
    as it writes. The alternative reads the stored value back and adds it in the
    accumulator, which costs one extra rounding to the output format

Every rounding step in that chain contributes one relative error term. K,
k_tiles_per_block, and math_fidelity set how many times each step happens;
math_fidelity and the two accumulation settings set how large each term is. The
terms are combined as the square root of the sum of their squares, the usual rule
for independent errors.

Stage two: the four limits, each built from "r" and the reference tensor, which is
needed whether or not an activation applies.

- atol. An element's error comes from rounding partial sums, whose size does not
  depend on the dot product they land in, so it is about the same for every
  element: "r" times the standard deviation of the pre-activation reference. One
  number therefore serves the whole tensor. It is a sum of many roundings, so it
  is close to Gaussian. An error can come out too high or too low, and the
  largest of n such errors, in either direction, sits about sqrt(2 * ln(2n))
  standard deviations from zero, for n the number of output elements. The module
  then shifts every reference value by that distance, up and down, applies the
  activation to all three values, and keeps the largest change in the
  activation's output. It does not scale the distance by the activation's
  slope, because at worst case distance the activation is nowhere near straight;
  this is why a saturating activation is never assigned more error than its whole
  output range. The device's own absolute error on the activation is then added.

- rtol takes the two contributions that do scale with an element's own value: the
  activation's relative error and the rounding into the output format. "r" does not
  enter it. The two are added rather than combined in quadrature, so that rtol
  stays a bound for every element rather than a typical value.

- frobenius_threshold combines three terms as the square root of the sum of their
  squares: "r" times the standard deviation of the pre-activation reference, scaled
  by the root-mean-square of the activation's slope; the output format rounding
  plus the activation's relative error, times the root-mean-square of the
  activated reference; and the device's absolute error on the activation. The
  result is divided by that same root-mean-square. A slope serves here, where
  atol needed an evaluation, because this term describes a typical error, and a
  slope averaged over an interval that size is the right thing for it.

- pcc_threshold reuses that combined error, before it is divided. Write q, the
  code's noise_ratio, for that error divided by the standard deviation of the
  activated reference, not its root-mean-square, because a correlation removes
  the mean of each tensor first. An error independent of the reference values adds
  to the device tensor's variance without adding to the covariance, which scales
  the correlation by 1 / sqrt(1 + q * q). matmul_numeric_tolerances returns the
  small q approximation of that, 1 - q * q / 2. One caveat: a uniform scaling
  leaves a correlation unchanged, so counting the truncation bias here asks for
  less correlation than is justified. That is safe against a false failure, but it
  lets more real error through.

atol, rtol and frobenius_threshold are multiplied by the safety argument, 2.0 by
default. pcc_threshold uses _PCC_MARGIN instead; see that constant for why.

To get limits for a new entry in a test's pytest.mark.parametrize list, build the
pre-activation product the test's reference is built from and call
matmul_numeric_tolerances with it. Run this from a Python session whose working
directory is the repository root, so that the import resolves:

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

Both per element limits can be checked by hand. For atol, "r" is 0.0284 here and
the pre-activation reference has a standard deviation of 22.6, so one element's
error has a standard deviation of 0.64. There are M * N = 65536 elements, and
sqrt(2 * ln(131072)) is 4.855, so the worst element error is 3.11. Reapplying
GELU either side of that shift raises it to 3.28, and the safety argument of 2.0
gives the printed 6.57. GELU contributes no absolute error, because the device
fits it with a polynomial and that inaccuracy counts as a relative error in rtol
instead. For rtol, the table gives GELU a relative error of 2 * 2**-7 = 0.0156 in
the accumulator format, bfloat16 here because fp32_dest_acc_en is False, and the
output rounding adds 2**-8 / sqrt(3) = 0.0023 (see _format_relative_error). Twice
their sum is the printed 0.0358.

One parametrize entry usually covers several shapes, data types and settings. Run
the calculation for each and keep the largest atol, rtol and
frobenius_threshold and the smallest pcc_threshold. Round the first three up and
pcc_threshold down.

Notes:

- The relative error does not grow with K. Each product carries a relative
rounding error from the narrowed mantissas, and the test inputs are a mix of
positive and negative numbers, so those errors multiply values that are as often
negative as positive and partly cancel. The total error grows as sqrt(K), the
result grows as sqrt(K), and the ratio is flat. The two terms that do grow
are the ones that round a running total: the accumulator (see
_K_PER_ACCUMULATION) and the partial sum spilled between K blocks. Both grow as
sqrt(K), never as K. With fp32_dest_acc_en both are held in 32 bit, so they fall
below every other term and K stops mattering; they matter only with a 16 bit
accumulator.

- Truncation is biased, and it still does not grow with K. The matrix engine
drops the low mantissa bits of each operand instead of rounding to nearest, which
moves that operand toward zero, so no product comes out larger in magnitude than
the true one. Write the product of a and b as a*b*(1-d), where d is the fraction
lost from the two truncated operands together; d is never negative. Only d has a
fixed sign, though. Each product a*b is positive or negative depending on its
operands, so the errors -d*a*b are too, and they partly cancel in the sum just as
the products themselves do.

The bias does decide how truncation enters r. An error that averages to zero
contributes its standard deviation divided by the result's. Truncation does not
average to zero, so _truncation_relative_error uses a root-mean-square instead,
which covers the average of d and the spread around it together. That is an upper
bound rather than an exact treatment of a fixed offset, and that function records
how much margin it leaves.
"""

import math

import torch
import ttnn

_SQRT3 = math.sqrt(3.0)
_SQRT6 = math.sqrt(6.0)

# Margin on the correlation limit, as a multiplier on the relative error, which
# the conversion to a correlation squares into 1.5x on the distance from 1. It is
# less than the safety factor the other three limits take, because the
# root-mean-square it is built from averages tens of thousands of elements and so
# barely moves from run to run. Some margin is still needed: the prediction
# tracks the measured error closely rather than sitting well above it.
_PCC_MARGIN = math.sqrt(1.5)

# Stored mantissa bits, excluding the leading one bit that normalised floating
# point formats imply rather than store. The gap between neighbouring values, one
# unit in the last place, is 2**-bits times the place value of that leading bit.
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

# Mantissa bits the matrix engine keeps from each operand, as (SrcA, SrcB), its
# two operand registers, taken from the union of the per-pass bit masks. The
# multiplier is 5 bits by 7 bits, and each fidelity level makes another pass to
# consume more of the inputs
# (tech_reports/matrix_engine/matrix_engine.md). A matmul loads the right hand
# operand into SrcA and the left hand operand into SrcB, the opposite of other
# operations, so the right hand operand takes the narrower path.
#
# Over an 11 bit significand field with the hidden bit at position 10, the masks
# cover SrcA 10..6 then 5..1, and SrcB 10..4 then 3..0. Accumulating those:
#   LoFi   pass 0        SrcA 4 bits,  SrcB 6 bits
#   HiFi2  passes 0,1    SrcA 9 bits,  SrcB 6 bits
#   HiFi3  passes 0,1,2  SrcA 9 bits,  SrcB 10 bits
#   HiFi4  all passes    same coverage as HiFi3
# Nine and ten bits exceed what bfloat16 and the block float formats carry, so
# those operands are consumed whole and contribute nothing.
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
    """Contribution to "r" from one rounding of a value into dtype.

    For a normalised format this is the rounding error relative to the value.
    For a block float format the step is fixed across a block of 16 by the
    largest magnitude in it, which for normally distributed data is about two
    standard deviations, so the error is absolute rather than relative. An error
    of k standard deviations on an operand enters the sum of squares as the same
    size of term a relative error of k would, which is why both cases return one
    comparable number.
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
    """Contribution to "r" from dropping mantissa bits, for one operand.

    The operand arrives with bits_available mantissa bits and the matrix engine
    keeps only bits_kept of them. If it keeps at least as many as arrive, nothing
    is dropped and the contribution is zero.

    The mantissa bits encode a significand, a value at least 1 and below 2.
    Dropping the bits below bits_kept removes less than the value of the lowest
    kept bit, one unit in the last kept place, u = 2**-bits_kept. Taking the
    amount removed as equally likely anywhere from zero to u, its average is u/2
    and its standard deviation is u/sqrt(12), so its root-mean-square is
    u*sqrt(1/4 + 1/12) = u/sqrt(3). The root-mean-square is the figure to use
    rather than the standard deviation, which would be half as large, because
    truncation is biased; see the module docstring.

    What "r" needs is the loss relative to the operand's own value. The amount
    removed and the value share the same power of two, so that factor cancels and
    the relative loss is the amount removed divided by the significand alone.
    Taking the two as independent, the mean square of the ratio is the mean square
    of the amount removed times the average of one over the square of the
    significand, and taking the significand as equally likely anywhere in its
    range that average is exactly a half: the area under 1/s**2 from 1 to 2 is
    1 - 1/2, over a range of width 1. So the answer is u/sqrt(3) times sqrt(1/2),
    which is u/sqrt(6). That same division turns the average u/2 into mu.

    Both uniform assumptions are approximations, so the formula is checked against
    measurement. Truncating bfloat16 values from torch.randn, which carry 7
    mantissa bits, and measuring the root-mean-square relative loss on one operand
    gives 0.0236, 0.0107 and 0.0041 for 4, 5 and 6 kept bits, that is 3, 2 and 1
    dropped bits, against this formula's 0.0255, 0.0128 and 0.0064: larger every
    time, by 8, 20 and 57 percent. The overstatement grows as fewer bits are
    dropped, because the amount removed then takes only a few discrete values
    instead of the continuous spread assumed above.

    An understatement in how the caller combines the operands nearly cancels that.
    A product loses from both operands, and matmul_relative_error adds the two
    contributions as the square root of the sum of squares, which drops the cross
    term between the two biases; for bfloat16 operands at LoFi that cross term is
    worth 16 percent. The two effects very nearly cancel there, leaving the caller
    0.5 percent above the measured pair, so this term carries almost no margin of
    its own and relies on the safety factor.
    """
    if bits_kept >= bits_available:
        return 0.0
    return 2.0**-bits_kept / _SQRT6


def _accumulated_rounding_relative_error(step_error, steps):
    """Contribution to "r" from rounding a running total steps times.

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
        math_fidelity: how many passes the matrix engine makes over the inputs.
        fp32_dest_acc_en: whether the accumulator is 32 bit rather than 16 bit.
        packer_l1_acc: whether partial sums are accumulated in place in memory
            instead of being written out and read back once per K block.
        k_tiles_per_block: how many 32 element slices of K are summed before the
            partial sum is written out. The default of 1 gives the most blocks
            and so the most partial sum roundings.
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
# so they carry no error beyond the rounding of the value they produce. relu
# needs no evaluation at all: the packer applies it while writing the value out.
_EXACT_ACTIVATIONS = frozenset(
    {
        ttnn.UnaryOpType.RELU,
        ttnn.UnaryOpType.RELU6,
        ttnn.UnaryOpType.HARDTANH,
    }
)

# Every other activation the matmul can fuse is a fitted polynomial or an
# exponential. The implementations record maximum errors between 0.28 and 1 unit
# in the last place of the format they run in, so 2 is used for all of them,
# double the worst recorded.
_FITTED_ACTIVATION_ULPS = 2.0

# gelu, tanh and sigmoid each accept a flag that replaces the fitted polynomial
# with a handful of straight line segments. The error is then the peak distance
# from those segments to the true function, which is absolute: unlike a rounding
# error it does not shrink as the output does.
#
# tanh's segments are written out in the kernel that loads them, 0.90625*x, then
# 0.09375*x + 0.8125, then 1, joining at x = 1 and x = 2. Their greatest distance
# from tanh is at the first join, where 0.90625 - tanh(1) = 0.145.
#
# gelu's and sigmoid's tables are packed constants with no arithmetic form beside
# them, so 0.13 is not obtained that way. It is what the vector unit's own sweep
# tests record for both, in
# tt_metal/tt-llk/tests/python_tests/test_eltwise_unary_sfpu.py. That makes it the
# only figure in this module measured from the implementation instead of worked
# out from a format width or a written algorithm, so a device error the sweep
# tests also miss would pass here too.
#
# These three numbers are large enough to matter. The safety factor doubles 0.13
# to 0.26, and sigmoid's output range is only 1.0, so a case with the flag set
# accepts a result wrong by a quarter of everything sigmoid can produce. Such a
# case checks that the op runs and is roughly right, not that it computes the
# function accurately.
_PIECEWISE_LINEAR_ABSOLUTE_ERROR = {
    ttnn.UnaryOpType.GELU: 0.13,
    ttnn.UnaryOpType.TANH: 0.15,
    ttnn.UnaryOpType.SIGMOID: 0.13,
}

# Index of the parameter that selects the piecewise linear table. For sigmoid it
# is the second parameter rather than the first.
_PIECEWISE_LINEAR_FLAG_INDEX = {
    ttnn.UnaryOpType.GELU: 0,
    ttnn.UnaryOpType.TANH: 0,
    ttnn.UnaryOpType.SIGMOID: 1,
}

# The root-mean-square error of a piecewise linear fit is below its peak, because
# the segments meet the curve at their ends. Half the peak is a rough stand-in
# for the whole-tensor checks; the peak itself is used elementwise.
_PIECEWISE_LINEAR_RMS_FRACTION = 0.5

# Most activations pick a higher accuracy inner algorithm when the accumulator is
# 32 bit. These two do not: they select it from a compile time setting that only
# the standalone activation op defines, so fused into a matmul they always take
# their 16 bit polynomial branch however the accumulator is configured.
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

    Pass the result straight into the assertion, with the unit in the last place
    check off because this module predicts no bound for it:

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
        safety: multiplier on the predicted error, applied to atol, rtol and
            frobenius_threshold, which all scale linearly with it. It is not
            applied to pcc_threshold, which takes _PCC_MARGIN instead; see that
            constant.

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

    # An element's error is set by the spread of the whole dot product, not by
    # the element's own value. A bias carries none of that error; it is left in
    # because the tests that use one keep it far smaller than the matmul result.
    error_before_activation = relative_error * pre_activation.std().item()

    # Largest error one element can plausibly receive, before the activation.
    extreme_value_factor = math.sqrt(2.0 * math.log(2 * max(pre_activation.numel(), 2)))
    extreme_error_before_activation = extreme_value_factor * error_before_activation

    if activation_fn is None:
        golden = pre_activation
        slope_rms = 1.0
        max_propagated_error = extreme_error_before_activation
    else:
        golden = activation_fn(pre_activation).float()
        # A symmetric difference at the scale of the error, rather than at a
        # vanishing scale, averages the slope over the interval the error can
        # move the value, so relu's kink gives a partial slope not an undefined
        # one.
        step = error_before_activation if error_before_activation > 0.0 else 1e-6
        slope = ((activation_fn(pre_activation + step) - activation_fn(pre_activation - step)) / (2.0 * step)).float()
        slope_rms = slope.pow(2).mean().sqrt().item()
        # Evaluated at the extreme rather than scaled by a slope, so a
        # saturating activation cannot be assigned more error than its range.
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

    # Every element is credited with the extreme error rather than only the
    # unluckiest one, so this is an upper bound. The activation's own error is
    # already a bound, so it is added rather than combined in quadrature.
    max_error = max_propagated_error + activation_absolute

    # rtol carries only what scales with an element's own value; atol carries the
    # accumulated error, which does not.
    tolerances = {
        "atol": safety * max_error,
        "rtol": safety * (output_relative + activation_relative),
        "frobenius_threshold": safety * rms_error / golden_rms if golden_rms > 0.0 else safety * rms_error,
    }

    # 1 - q**2 / 2 for q the error over the activated reference's standard
    # deviation, not its root-mean-square: a correlation removes the mean first,
    # and for a saturating activation most of that size is in the mean. Capped at
    # 0.999999, since no test should demand more agreement than that. An all-zero
    # reference has no scale to divide by, so the frobenius limit above is left
    # absolute and the correlation limit drops to zero.
    if golden_std > 0.0:
        noise_ratio = _PCC_MARGIN * rms_error / golden_std
        tolerances["pcc_threshold"] = max(0.0, min(0.999999, 1.0 - noise_ratio * noise_ratio / 2.0))
    else:
        tolerances["pcc_threshold"] = 0.0

    return tolerances
