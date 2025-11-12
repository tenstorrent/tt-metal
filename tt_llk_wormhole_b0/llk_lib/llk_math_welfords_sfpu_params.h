// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <algorithm>
#include <utility>

#include "llk_math_eltwise_ternary_sfpu.h"
#include "llk_sfpu_types.h"

template <typename Callable, typename... ARGS>
inline void _llk_math_welfords_sfpu_params_(Callable&& sfpu_func, uint dst_index0, ARGS&&... args)
{
    _llk_math_eltwise_ternary_sfpu_start_<DST_SYNC_MODE>(dst_index0);
    std::forward<Callable>(sfpu_func)(std::forward<ARGS>(args)...);
    _llk_math_eltwise_ternary_sfpu_done_(); // Finalize
}
