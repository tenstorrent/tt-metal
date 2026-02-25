// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_common_api.h"
#include "../../../../../../tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_deepseek_moe_gate_transpose_dest_single_face.h"

namespace ckernel {

template <bool is_32bit = false>
inline void llk_math_deepseek_moe_gate_transpose_dest_single_face_common_init() {
    _llk_math_deepseek_moe_gate_transpose_dest_single_face_common_init_<is_32bit>();
}

template <bool is_32bit = false>
inline void llk_math_deepseek_moe_gate_transpose_dest_single_face_step0_init() {
    _llk_math_deepseek_moe_gate_transpose_dest_single_face_step0_init_<is_32bit>();
}

template <bool is_fp32_dest_acc_en, bool is_32bit = false>
inline void llk_math_deepseek_moe_gate_transpose_dest_single_face_step0() {
    _llk_math_deepseek_moe_gate_transpose_dest_single_face_step0_<is_fp32_dest_acc_en, is_32bit>();
}

template <bool is_32bit = false>
inline void llk_math_deepseek_moe_gate_transpose_dest_single_face_step1_init() {
    _llk_math_deepseek_moe_gate_transpose_dest_single_face_step1_init_<is_32bit>();
}

template <bool is_fp32_dest_acc_en, bool is_32bit = false>
inline void llk_math_deepseek_moe_gate_transpose_dest_single_face_step1() {
    _llk_math_deepseek_moe_gate_transpose_dest_single_face_step1_<is_fp32_dest_acc_en, is_32bit>();
}

template <bool is_32bit = false>
inline void llk_math_deepseek_moe_gate_transpose_dest_single_face_step2_init() {
    _llk_math_deepseek_moe_gate_transpose_dest_single_face_step2_init_<is_32bit>();
}

template <bool is_fp32_dest_acc_en, bool is_32bit = false>
inline void llk_math_deepseek_moe_gate_transpose_dest_single_face_step2() {
    _llk_math_deepseek_moe_gate_transpose_dest_single_face_step2_<is_fp32_dest_acc_en, is_32bit>();
}

}  // namespace ckernel
