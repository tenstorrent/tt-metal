// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_generalized_moe_gate_topk_single_face.h"

namespace ckernel {

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_sfpu_generalized_moe_gate_topk_init() {
    // Don't need the second addrmod so set type to unused
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(
        sfpu::generalized_moe_gate_topk_init<APPROXIMATE, is_fp32_dest_acc_en>);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_sfpu_generalized_moe_gate_sum_top2(
    uint dst_index, VectorMode vector_mode = VectorMode::RC_custom) {
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::generalized_moe_gate_sum_top2<APPROXIMATE, is_fp32_dest_acc_en>, dst_index, vector_mode);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_sfpu_generalized_moe_gate_sort_top4_groups(
    uint dst_index, VectorMode vector_mode = VectorMode::RC_custom) {
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::generalized_moe_gate_sort_top4_groups<APPROXIMATE, is_fp32_dest_acc_en>, dst_index, vector_mode);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_sfpu_generalized_moe_gate_top8(
    uint dst_index, uint32_t eps, uint32_t scale, VectorMode vector_mode = VectorMode::RC_custom) {
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::generalized_moe_gate_top8<APPROXIMATE, is_fp32_dest_acc_en>, dst_index, vector_mode, eps, scale);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en, uint32_t read_base, uint32_t store_lo, uint32_t store_hi>
inline void llk_math_sfpu_generalized_moe_gate_merge4_top8(
    uint dst_index, VectorMode vector_mode = VectorMode::RC_custom) {
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::
            generalized_moe_gate_merge4_top8<APPROXIMATE, is_fp32_dest_acc_en, read_base, store_lo, store_hi>,
        dst_index,
        vector_mode);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en, uint32_t store_lo, uint32_t store_hi, uint32_t idx_offset>
inline void llk_math_sfpu_generalized_moe_gate_merge16_to_run(
    uint dst_index, VectorMode vector_mode = VectorMode::RC_custom) {
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::
            generalized_moe_gate_merge16_to_run<APPROXIMATE, is_fp32_dest_acc_en, store_lo, store_hi, idx_offset>,
        dst_index,
        vector_mode);
}

template <
    bool APPROXIMATE,
    bool is_fp32_dest_acc_en,
    uint32_t from_lo,
    uint32_t from_hi,
    uint32_t to_lo,
    uint32_t to_hi>
inline void llk_math_sfpu_generalized_moe_gate_copy_topk_run(
    uint dst_index, VectorMode vector_mode = VectorMode::RC_custom) {
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::
            generalized_moe_gate_copy_topk_run<APPROXIMATE, is_fp32_dest_acc_en, from_lo, from_hi, to_lo, to_hi>,
        dst_index,
        vector_mode);
}

template <
    bool APPROXIMATE,
    bool is_fp32_dest_acc_en,
    uint32_t field,
    uint32_t src_lo,
    uint32_t src_hi,
    uint32_t dst_lo,
    uint32_t dst_hi>
inline void llk_math_sfpu_generalized_moe_gate_place_field_from_interm(
    uint dst_index, VectorMode vector_mode = VectorMode::RC_custom) {
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::generalized_moe_gate_place_field_from_interm<
            APPROXIMATE,
            is_fp32_dest_acc_en,
            field,
            src_lo,
            src_hi,
            dst_lo,
            dst_hi>,
        dst_index,
        vector_mode);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en, uint32_t topk = 8, bool output_softmax = false>
inline void llk_math_sfpu_generalized_moe_gate_finalize_ungrouped(
    uint dst_index, uint32_t eps, uint32_t scale, VectorMode vector_mode = VectorMode::RC_custom) {
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::generalized_moe_gate_finalize_ungrouped<APPROXIMATE, is_fp32_dest_acc_en, topk, output_softmax>,
        dst_index,
        vector_mode,
        eps,
        scale);
}

}  // namespace ckernel
