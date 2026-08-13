// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Test-local operation selector. Some SFPI kernels already exist on Quasar but
// do not yet have a corresponding entry in Quasar's production SfpuType enum.
// Keeping the selector in the test tree lets those kernels be exercised without
// changing production kernel interfaces merely to support a test harness.
enum class QuasarSfpuTestOperation
{
    abs,
    square,
    rsqrt,
    exponential,
    gelu,
    relu,
    reciprocal,
    sqrt,
    sqrt_custom,
    log,
    log1p,
    tanh,
    sigmoid,
    silu,
    clamp,
    negative,
    softplus,
    typecast,
    sine,
    cosine,
    tan,
    atan,
    asin,
    acos,
    sinh,
    cosh,
    acosh,
    asinh,
    atanh,
    equal_zero,
    not_equal_zero,
    less_than_zero,
    greater_than_zero,
    less_than_equal_zero,
    greater_than_equal_zero,
};
