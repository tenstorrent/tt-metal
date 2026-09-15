// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/experimental/ckernel_sfpu_softmax_k.h"

namespace ckernel {
namespace sfpu {

inline void llk_math_sfpu_softmax_k_init() { llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(_init_softmax_k_); }

template <int k>
inline void llk_math_sfpu_softmax_k() {
    _llk_math_eltwise_unary_sfpu_params_(_softmax_k_<k>, 0, VectorMode::RC_custom);
}

}  // namespace sfpu
}  // namespace ckernel
