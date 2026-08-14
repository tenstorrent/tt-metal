# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fixed scalar dispatch constants, mirrored once from sfpu_operations.h.

These are the numbers a kernel bakes in: clamp's bounds, softplus' linear threshold, the
scalar the integer max/min compares against. Two consumers need each of them, for
different reasons:

  * ``golden_generators.UnarySFPUGolden`` reproduces the value so its result matches what
    the kernel computes.
  * ``sfpu_domains._OP_EDGE_POINTS`` probes *exactly at* the value, because that is where
    the op's behaviour switches — the knee, the threshold, the comparison tie.

Held separately because two independent copies of one number is a silent-drift bug rather
than duplication: raise the golden's threshold and an edge table that restates it keeps
probing a point that is no longer a threshold, which reads as full coverage while testing
nothing. Neither consumer owns the number, so neither imports it from the other.

This module is the leaf of that dependency: it must not import from golden_generators or
sfpu_domains.
"""

# Comparison ops (UnaryGt/Lt/Ge/Le/Eq/Ne) compare x against this.
UNARY_COMP_THRESHOLD = 0.5

# UnaryMax / UnaryMin compare x against this.
UNARY_MAX_MIN_VALUE = 0.0

# clamp / hardtanh bounds (with offset 0).
CLAMP_MIN = -1.0
CLAMP_MAX = 1.0

# Shrinkage lambdas: the value below which the op returns 0.
SOFTSHRINK_LAMBDA = 0.5
HARDSHRINK_LAMBDA = 0.5

# softplus(x) = log1p(exp(beta*x))/beta, going linear above the threshold.
SOFTPLUS_BETA = 1.0
SOFTPLUS_THRESHOLD = 20.0

# threshold(x): below T the output jumps to V.
THRESHOLD_T = 5.0
THRESHOLD_V = 10.0

# relu_max clamps above at this value; relu_min clamps below at it.
RELU_MAX_THRESHOLD = 5.0
RELU_MIN_THRESHOLD = 5.0

# Negative-side slopes.
PRELU_SLOPE = 0.25
LRELU_NEGATIVE_SLOPE = 0.1

# UnaryMax/MinInt32 and UnaryMax/MinUint32 compare against this scalar.
INT_MAXMIN_SCALAR = 1000

# Unary bitwise (calculate_sfpu_unary_bitwise) masks against this scalar. Chosen with bits
# set in both halves of the word and neither all-ones nor a single bit, so AND, OR and XOR
# each produce a value distinguishable from the input and from each other.
UNARY_BITWISE_SCALAR = 0x0F0F0F0F

# Unary left_shift / right_shift shift by this many bits.
UNARY_SHIFT_AMOUNT = 3

# Unary fmod / remainder divide by this fixed divisor.
UNARY_MOD_DIVISOR = 2.0

# heaviside(x) returns this when x == 0.
HEAVISIDE_VALUE = 0.5

# celu / elu negative-branch alpha.
ELU_ALPHA = 1.0
CELU_ALPHA = 1.0

# selu's fixed scale and alpha (0x3f867d5f / 0x3fd62d7d as fp32 bit patterns).
SELU_SCALE = 1.0507009873554805
SELU_ALPHA = 1.6732632423543772
