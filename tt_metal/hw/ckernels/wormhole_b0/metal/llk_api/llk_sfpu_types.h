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
    softcap,
    // The following are referenced by third-party LLK template bodies
    // and must exist to avoid -Wtemplate-body errors, even though they
    // are never instantiated in this build.
    equal_zero,
    not_equal_zero,
    less_than_zero,
    greater_than_equal_zero,
    greater_than_zero,
    less_than_equal_zero,
    topk_local_sort,
    typecast,
    isfinite,
    isinf,
    isnan,
    isposinf,
    isneginf,
    signbit,
    unary_eq,
    unary_ne,
    unary_gt,
    unary_lt,
    unary_ge,
    unary_le,
    unary_max,
    unary_min,
    unary_max_int32,
    unary_min_int32,
    unary_max_uint32,
    unary_min_uint32,
};
