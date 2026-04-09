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
    // Values below are referenced by LLK template code but not used by metal SFPU ops.
    // They must exist so the JIT compiler can parse template bodies with if-constexpr guards.
    equal_zero,
    not_equal_zero,
    less_than_zero,
    greater_than_equal_zero,
    less_than_equal_zero,
    greater_than_zero,
    topk_local_sort,
    typecast,
    unary_max,
    unary_min,
    unary_max_int32,
    unary_min_int32,
    unary_max_uint32,
    unary_min_uint32,
    signbit,
    unary_ne,
    unary_eq,
    unary_gt,
    unary_lt,
    unary_ge,
    unary_le,
    isinf,
    isposinf,
    isneginf,
    isnan,
    isfinite,
    mul_int32,
    mul_uint16,
    max,
    min,
    max_int32,
    min_int32,
    max_uint32,
    min_uint32,
    where,
};
