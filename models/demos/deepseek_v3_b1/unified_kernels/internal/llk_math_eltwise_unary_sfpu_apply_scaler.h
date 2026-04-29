// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// !!! INTERNAL — deepseek_v3_b1 only !!!
// LLK wrapper for the deepseek-private apply_scaler SFPU op. NOT exposed via
// llk_math_unary_sfpu_api.h; do not pull this header from outside the
// deepseek_v3_b1 tree. Uses SfpuType::unused so it doesn't claim a slot in
// the shared SFPU op enum (llk_sfpu_types.h).

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_apply_scaler.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_apply_scaler_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(sfpu::apply_scaler_init<APPROXIMATE>);
}

// Wrapper template params kept only for API parity with the standard sigmoid/
// silu wrappers (so call sites don't have to special-case this op). The body
// passes calculate_apply_scaler as a plain function pointer — the op itself
// has no compile-time knobs.
template <bool APPROXIMATE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_apply_scaler(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::calculate_apply_scaler, dst_index, vector_mode);
}

}  // namespace ckernel
