// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "experimental/llk_math_fast_untilize.h"

/*************************************************************************
 * LLK MATH FAST UNTILIZE (BH)
 *************************************************************************/

inline void llk_math_fast_untilize_init() { ckernel::_llk_math_fast_untilize_init_(); }

inline void llk_math_fast_untilize_init_skip_remap() { ckernel::_llk_math_fast_untilize_init_<false>(); }

inline void llk_math_fast_untilize_block(const std::uint32_t dst_index, const std::uint32_t block_ct_dim) {
    ckernel::_llk_math_fast_untilize_block_<DST_ACCUM_MODE>(dst_index, block_ct_dim);
}

inline void llk_math_fast_untilize_uninit() { ckernel::_llk_math_fast_untilize_uninit_(); }
