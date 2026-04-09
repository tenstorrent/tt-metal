// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

enum class SfpuType {
    unused = 0,
    frac,
    swish,
    atanh,
    sinh,
    rrelu,
    // Comparison types used by third-party LLK
    equal_zero,
    not_equal_zero,
    less_than_zero,
    greater_than_equal_zero,
    greater_than_zero,
    less_than_equal_zero,
    // Unary comparison types
    unary_ne,
    unary_eq,
    unary_gt,
    unary_lt,
    unary_ge,
    unary_le,
    // Inf/NaN types
    isinf,
    isposinf,
    isneginf,
    isnan,
    isfinite,
};
