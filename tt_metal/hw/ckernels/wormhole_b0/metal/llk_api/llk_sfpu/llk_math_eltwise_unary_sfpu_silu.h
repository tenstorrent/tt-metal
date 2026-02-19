// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_silu.h"
#include "vconst_verifier.h"

namespace ckernel {

template <bool APPROXIMATE, typename vConstVerifier = vconst_verifier::disable>
inline auto llk_math_eltwise_unary_sfpu_silu_init() {
    return llk_math_eltwise_unary_sfpu_init<SfpuType::silu, APPROXIMATE>(sfpu::silu_init<APPROXIMATE, vConstVerifier>);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en, typename vConstVerifier = vconst_verifier::disable>
inline auto llk_math_eltwise_unary_sfpu_silu(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    return _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_silu<is_fp32_dest_acc_en, 8, vConstVerifier>, dst_index, vector_mode);
}

}  // namespace ckernel
