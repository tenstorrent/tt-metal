// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_ternary_sfpu.h"

namespace ckernel {

inline void llk_math_ema_sfpu_init() { _llk_math_eltwise_ternary_sfpu_init_<SfpuType::unused>(); }

inline void llk_math_ema_sfpu_load_alpha_beta(uint32_t alpha, uint32_t beta) { sfpu::_load_alpha_beta_(alpha, beta); }

inline void llk_math_ema_sfpu_clear_previous_output() { sfpu::_clear_previous_output_(); }

inline void llk_math_ema_sfpu_tile(uint32_t input_dst_index) {
    _llk_math_eltwise_ternary_sfpu_start_<DST_SYNC_MODE>(input_dst_index);
    sfpu::_calculate_ema_tile_();
    _llk_math_eltwise_ternary_sfpu_done_();
}

}  // namespace ckernel
