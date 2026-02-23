// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "../../../../../../../tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_deepseek_moe_gate_topk_single_face.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <ckernel::ApproximationMode APPROX_MODE, bool is_fp32_dest_acc_en>
inline void deepseek_moe_gate_sum_top2() {
    _deepseek_moe_gate_sum_top2<APPROX_MODE, is_fp32_dest_acc_en>();
}

template <ckernel::ApproximationMode APPROX_MODE, bool is_fp32_dest_acc_en>
inline void deepseek_moe_gate_sort_top4_groups() {
    _deepseek_moe_gate_sort_top4_groups<APPROX_MODE, is_fp32_dest_acc_en>();
}

template <ckernel::ApproximationMode APPROX_MODE, bool is_fp32_dest_acc_en>
inline void deepseek_moe_gate_top8(uint32_t eps, uint32_t scale) {
    _deepseek_moe_gate_top8<APPROX_MODE, is_fp32_dest_acc_en>(eps, scale);
}

template <ckernel::ApproximationMode APPROX_MODE, bool is_fp32_dest_acc_en>
inline void deepseek_moe_gate_topk_init() {
    _init_deepseek_moe_gate_topk<APPROX_MODE, is_fp32_dest_acc_en>();
}

}  // namespace sfpu
}  // namespace ckernel
