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

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void deepseek_moe_gate_sum_top2(uint32_t dst_index_in, uint32_t dst_index_out) {
    _deepseek_moe_gate_sum_top2<APPROXIMATION_MODE, is_fp32_dest_acc_en>(dst_index_in, dst_index_out);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void deepseek_moe_gate_sort_top4_groups(uint32_t dst_index_in, uint32_t dst_index_out) {
    _deepseek_moe_gate_sort_top4_groups<APPROXIMATION_MODE, is_fp32_dest_acc_en>(dst_index_in, dst_index_out);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void deepseek_moe_gate_top8(uint32_t dst_index_in, uint32_t dst_index_out, uint32_t eps, uint32_t scale) {
    _deepseek_moe_gate_top8<APPROXIMATION_MODE, is_fp32_dest_acc_en>(dst_index_in, dst_index_out, eps, scale);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void deepseek_moe_gate_topk_init() {
    _init_deepseek_moe_gate_topk<APPROXIMATION_MODE, is_fp32_dest_acc_en>();
}

}  // namespace sfpu
}  // namespace ckernel
