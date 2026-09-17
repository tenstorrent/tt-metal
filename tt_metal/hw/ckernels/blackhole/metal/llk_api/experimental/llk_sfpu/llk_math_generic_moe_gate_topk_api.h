// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/experimental/ckernel_sfpu_generic_moe_gate_topk.h"
#include "sanitizer/api.h"

namespace ckernel {
namespace sfpu {

inline void llk_math_sfpu_generic_moe_gate_topk_init() {
    SAN_HOOK(unsupported());
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(_init_generic_moe_gate_topk_);
}

template <
    bool normalize,
    int num_selected_experts,
    int num_total_experts,
    bool zero_tail = false,
    bool full_sort = false,
    bool generate_indices = true,
    bool do_extra_scale = false>
inline void llk_math_sfpu_generic_moe_gate_topk(uint32_t eps, uint32_t scale, uint32_t extra_scale = 0) {
    SAN_HOOK(unsupported());
    _llk_math_eltwise_unary_sfpu_params_(
        _generic_moe_gate_topk_<
            normalize,
            num_selected_experts,
            num_total_experts,
            zero_tail,
            full_sort,
            generate_indices,
            do_extra_scale>,
        0,
        VectorMode::RC_custom,
        eps,
        scale,
        extra_scale);
}

}  // namespace sfpu
}  // namespace ckernel
