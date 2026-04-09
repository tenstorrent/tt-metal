// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
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
    // Stubs required by third-party LLK headers (-Wtemplate-body)
    topk_local_sort,
    typecast,
    signbit,
    where,
    // Comparisons
    equal_zero,
    not_equal_zero,
    less_than_zero,
    greater_than_equal_zero,
    greater_than_zero,
    less_than_equal_zero,
    unary_ne,
    unary_eq,
    unary_gt,
    unary_lt,
    unary_ge,
    unary_le,
    // Special value checks
    isinf,
    isposinf,
    isneginf,
    isnan,
    isfinite,
    // Binary SFPU
    mul_int32,
    mul_uint16,
    max,
    min,
    max_int32,
    min_int32,
    max_uint32,
    min_uint32,
    // Unary min/max
    unary_max,
    unary_min,
    unary_max_int32,
    unary_min_int32,
    unary_max_uint32,
    unary_min_uint32,
};
