// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_common_api.h"
#include "experimental/llk_math_generalized_moe_gate_transpose_dest_single_face.h"

namespace ckernel {

template <bool is_32bit = false>
inline void llk_math_generalized_moe_gate_transpose_dest_single_face_common_init() {
    _llk_math_generalized_moe_gate_transpose_dest_single_face_common_init_<is_32bit>();
}

template <bool is_32bit = false>
inline void llk_math_generalized_moe_gate_transpose_dest_single_face_step0_init() {
    _llk_math_generalized_moe_gate_transpose_dest_single_face_step0_init_<is_32bit>();
}

template <bool is_fp32_dest_acc_en, bool is_32bit = false>
inline void llk_math_generalized_moe_gate_transpose_dest_single_face_step0() {
    _llk_math_generalized_moe_gate_transpose_dest_single_face_step0_<is_fp32_dest_acc_en, is_32bit>();
}

template <bool is_32bit = false>
inline void llk_math_generalized_moe_gate_transpose_dest_single_face_step1_init() {
    _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_init_<is_32bit>();
}

template <bool is_fp32_dest_acc_en, bool is_32bit = false>
inline void llk_math_generalized_moe_gate_transpose_dest_single_face_step1() {
    _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_<is_fp32_dest_acc_en, is_32bit>();
}

template <std::uint32_t src = 0, std::uint32_t dst = 0, bool is_32bit = false, std::uint32_t srcb = 16>
inline void llk_math_generalized_moe_gate_copy4rows_init() {
    _llk_math_generalized_moe_gate_copy4rows_init_<src, dst, is_32bit, srcb>();
}

template <bool is_fp32_dest_acc_en, bool is_32bit = false>
inline void llk_math_generalized_moe_gate_copy4rows() {
    _llk_math_generalized_moe_gate_copy4rows_<is_fp32_dest_acc_en, is_32bit>();
}

template <std::uint32_t d2b_dst = 0, std::uint32_t b2d_base = 24, bool is_32bit = false>
inline void llk_math_generalized_moe_gate_transpose_dest_single_face_step1_hi_init() {
    _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_hi_init_<d2b_dst, b2d_base, is_32bit>();
}

template <bool is_fp32_dest_acc_en, bool is_32bit = false>
inline void llk_math_generalized_moe_gate_transpose_dest_single_face_step1_hi() {
    _llk_math_generalized_moe_gate_transpose_dest_single_face_step1_hi_<is_fp32_dest_acc_en, is_32bit>();
}

template <bool is_32bit = false>
inline void llk_math_generalized_moe_gate_transpose_dest_single_face_step2_init() {
    _llk_math_generalized_moe_gate_transpose_dest_single_face_step2_init_<is_32bit>();
}

template <bool is_fp32_dest_acc_en, bool is_32bit = false>
inline void llk_math_generalized_moe_gate_transpose_dest_single_face_step2() {
    _llk_math_generalized_moe_gate_transpose_dest_single_face_step2_<is_fp32_dest_acc_en, is_32bit>();
}

}  // namespace ckernel
