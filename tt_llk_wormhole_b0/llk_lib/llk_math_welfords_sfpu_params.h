// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <algorithm>
#include <utility>

#include "llk_math_eltwise_ternary_sfpu.h"
#include "llk_sfpu_types.h"

template <typename Callable, typename... ARGS>
inline void _llk_math_welfords_sfpu_params_(Callable&& sfpu_func, uint dst_index0, uint dst_index1, uint dst_index2, ARGS&&... args)
{
    // Compute minimum destination index
    // not sure why we use the min. But this LLK is designed such that
    //
    // dst_index0 is the input tile (TILIZED FORMAT ONLY)
    // dst_index1 is the previous mean. It will be zero otherwise
    // dst_index1 is the previous var
    //
    uint dst_index = std::min(dst_index0, std::min(dst_index1, dst_index2));
    _llk_math_eltwise_ternary_sfpu_start_<DST_SYNC_MODE>(dst_index); // Reuse same sync primitive
    std::forward<Callable>(sfpu_func)(std::forward<ARGS>(args)...);  // Need to replace the above line with this
    _llk_math_eltwise_ternary_sfpu_done_();                          // Finalize
}
