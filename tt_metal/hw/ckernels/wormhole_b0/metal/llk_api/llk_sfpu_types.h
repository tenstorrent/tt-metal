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
    // Members referenced by tt_llk submodule code (ckernel_sfpu_comp.h, etc.)
    equal_zero,
    not_equal_zero,
    less_than_zero,
    greater_than_equal_zero,
    less_than_equal_zero,
    greater_than_zero,
    topk_local_sort,
    typecast,
    signbit,
    isfinite,
    isinf,
    isnan,
    isposinf,
    isneginf,
    unary_ne,
    unary_eq,
    unary_gt,
    unary_lt,
    unary_ge,
    unary_le,
    unary_max_int32,
    unary_min_int32,
    min,
    min_uint32,
    unary_max,
    unary_min,
    unary_max_uint32,
    unary_min_uint32,
    where,
};
