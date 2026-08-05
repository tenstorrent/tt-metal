// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/experimental/ckernel_sfpu_generic_moe_gate_topk.h"

namespace ckernel {
namespace sfpu {

inline void llk_math_sfpu_generic_moe_gate_topk_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(_init_generic_moe_gate_topk_);
}

template <
    bool normalize,
    int num_selected_experts,
    int num_total_experts,
    bool zero_tail = false,
    bool full_sort = false,
    bool generate_indices = true>
inline void llk_math_sfpu_generic_moe_gate_topk(uint32_t eps, uint32_t scale) {
    _llk_math_eltwise_unary_sfpu_params_(
        _generic_moe_gate_topk_<
            normalize,
            num_selected_experts,
            num_total_experts,
            zero_tail,
            full_sort,
            generate_indices>,
        0,
        VectorMode::RC_custom,
        eps,
        scale);
}

}  // namespace sfpu
}  // namespace ckernel
