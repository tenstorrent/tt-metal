// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/experimental/ckernel_sfpu_softmax_k.h"
#include "sanitizer/api.h"

namespace ckernel {
namespace sfpu {

template <bool is_fp32_dest_acc_en>
inline void llk_math_sfpu_softmax_k_init() {
    SAN_HOOK(unsupported());
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(_init_softmax_k_<is_fp32_dest_acc_en>);
}

template <int k, bool is_fp32_dest_acc_en>
inline void llk_math_sfpu_softmax_k() {
    SAN_HOOK(unsupported());
    _llk_math_eltwise_unary_sfpu_params_(_softmax_k_<k, is_fp32_dest_acc_en>, 0, VectorMode::RC_custom);
}

}  // namespace sfpu
}  // namespace ckernel
