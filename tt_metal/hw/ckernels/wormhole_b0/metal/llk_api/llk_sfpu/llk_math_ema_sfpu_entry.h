// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_sfpu_types.h"
#include "llk_math_ema_sfpu.h"
#include "llk_math_eltwise_ternary_sfpu.h"

namespace ckernel {

inline void llk_math_ema_sfpu_init() { _llk_math_ema_sfpu_init_(); }

template <uint32_t input_dst_index>
inline void llk_math_ema_sfpu(bool first_sample) {
    _llk_math_eltwise_ternary_sfpu_start_<DST_SYNC_MODE>(input_dst_index);
    ckernel::sfpu::_calculate_ema_online_(first_sample);
    _llk_math_eltwise_ternary_sfpu_done_();  // Finalize
}

}  // namespace ckernel
