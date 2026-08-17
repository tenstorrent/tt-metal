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
