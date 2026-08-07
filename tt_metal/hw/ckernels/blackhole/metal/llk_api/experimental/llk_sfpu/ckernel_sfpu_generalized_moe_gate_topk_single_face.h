// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpu/experimental/ckernel_sfpu_generalized_moe_gate_topk_single_face.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void generalized_moe_gate_sum_top2() {
    _generalized_moe_gate_sum_top2<APPROXIMATION_MODE, is_fp32_dest_acc_en>();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void generalized_moe_gate_sort_top4_groups() {
    _generalized_moe_gate_sort_top4_groups<APPROXIMATION_MODE, is_fp32_dest_acc_en>();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void generalized_moe_gate_top8(uint32_t eps, uint32_t scale) {
    _generalized_moe_gate_top8<APPROXIMATION_MODE, is_fp32_dest_acc_en>(eps, scale);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, uint32_t read_base, uint32_t store_lo, uint32_t store_hi>
inline void generalized_moe_gate_merge4_top8() {
    _gmg_merge4_top8<is_fp32_dest_acc_en, read_base, store_lo, store_hi>();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, uint32_t store_lo, uint32_t store_hi, uint32_t idx_offset>
inline void generalized_moe_gate_merge16_to_run() {
    _gmg_merge16_to_run<APPROXIMATION_MODE, is_fp32_dest_acc_en, store_lo, store_hi, idx_offset>();
}

template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en,
    uint32_t from_lo,
    uint32_t from_hi,
    uint32_t to_lo,
    uint32_t to_hi>
inline void generalized_moe_gate_copy_topk_run() {
    _gmg_copy_topk_run<from_lo, from_hi, to_lo, to_hi>();
}

template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en,
    uint32_t field,
    uint32_t src_lo,
    uint32_t src_hi,
    uint32_t dst_lo,
    uint32_t dst_hi>
inline void generalized_moe_gate_place_field_from_interm() {
    _gmg_place_field_from_interm<field, src_lo, src_hi, dst_lo, dst_hi>();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, uint32_t topk = 8, bool output_softmax = false>
inline void generalized_moe_gate_finalize_ungrouped(uint32_t eps, uint32_t scale) {
    _generalized_moe_gate_finalize_ungrouped<APPROXIMATION_MODE, is_fp32_dest_acc_en, topk, output_softmax>(eps, scale);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void generalized_moe_gate_topk_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    _init_generalized_moe_gate_topk<APPROXIMATION_MODE, is_fp32_dest_acc_en>();
}

}  // namespace sfpu
}  // namespace ckernel
