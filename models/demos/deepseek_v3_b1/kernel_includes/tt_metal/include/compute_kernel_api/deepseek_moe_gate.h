// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/transpose_wh.h"
#ifdef TRISC_MATH
#include "../../hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_deepseek_moe_gate_topk_single_face.h"
#include "../../hw/ckernels/blackhole/metal/llk_api/llk_math_deepseek_moe_gate_eltwise_binary_api.h"
#include "../../hw/ckernels/blackhole/metal/llk_api/llk_math_deepseek_moe_gate_transpose_dest_single_face_api.h"
#endif

namespace ckernel {

template <bool enable_sigmoid = false, bool is_32bit = false>
ALWI void deepseek_moe_gate_init(uint32_t icb0, uint32_t icb1) {
    if constexpr (enable_sigmoid) {
        // Init sigmoid (SFPU)
        sigmoid_tile_init<false>();
        // Init transpose wh (FPU)
        transpose_wh_init_short(icb0);
    } else {
        // Init copy add (FPU)
        UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(icb0, icb1, 1)));
        MATH((llk_math_deepseek_moe_gate_eltwise_binary_init_with_operands<
              ELWADD,
              DeepseekMoeGateEltwiseBinaryMode::COPY,
              MATH_FIDELITY>(icb0, icb1, false)));
        // Init transpose dest addrmods (does not conflict with copy add)
        MATH((llk_math_deepseek_moe_gate_transpose_dest_single_face_common_init<is_32bit>()));
        // Init topk (SFPU)
        MATH((llk_math_sfpu_deepseek_moe_gate_topk_init<APPROX, DST_ACCUM_MODE>()));
    }
}

template <bool enable_sigmoid = false, bool is_32bit = false>
ALWI void deepseek_moe_gate(uint32_t icb0, uint32_t icb1, uint32_t eps, uint32_t scale) {
    if constexpr (enable_sigmoid) {
        // Transpose wh (FPU)
        transpose_wh_tile(icb0, 0, 0);
        // Sigmoid (SFPU)
        sigmoid_tile<VectorMode::RC_custom, false>(0);
        // Init add binary reuse (FPU)
        UNPACK((llk_unpack_A_init<BroadcastType::NONE, true, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
            false, false, icb1)));
        MATH((llk_math_deepseek_moe_gate_eltwise_binary_init_with_operands<
              ELWADD,
              DeepseekMoeGateEltwiseBinaryMode::RELOAD,
              MATH_FIDELITY>(icb1, icb1, false)));
        // Add binary reuse (FPU)
        UNPACK((llk_unpack_A<BroadcastType::NONE, true, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(icb1, 0)));
        MATH((llk_math_deepseek_moe_gate_eltwise_binary<ELWADD, DST_ACCUM_MODE, MATH_FIDELITY>(icb1, icb1, 0, true)));
        // Init transpose dest addrmods (does not conflict with add binary reuse)
        MATH((llk_math_deepseek_moe_gate_transpose_dest_single_face_common_init<is_32bit>()));
        // Init topk (SFPU)
        MATH((llk_math_sfpu_deepseek_moe_gate_topk_init<APPROX, DST_ACCUM_MODE>()));
    } else {
        // Copy add (FPU)
        UNPACK((llk_unpack_AB(icb0, icb1, 0, 0)));
        MATH((llk_math_deepseek_moe_gate_eltwise_binary<ELWADD, DST_ACCUM_MODE, MATH_FIDELITY>(icb0, icb1, 0, true)));
    }
    // Set srcb dummy valid for transpose wh (FPU)
    UNPACK((llk_unpack_set_srcb_dummy_valid()));
    // Sum top2 (SFPU)
    MATH((llk_math_sfpu_deepseek_moe_gate_sum_top2<APPROX, DST_ACCUM_MODE>(0)));
    // Transpose dest step 0 (FPU)
    MATH((llk_math_deepseek_moe_gate_transpose_dest_single_face_step0_init<is_32bit>()));
    MATH((llk_math_deepseek_moe_gate_transpose_dest_single_face_step0<DST_ACCUM_MODE, is_32bit>()));
    // Sort top4 groups (SFPU)
    MATH((llk_math_sfpu_deepseek_moe_gate_sort_top4_groups<APPROX, DST_ACCUM_MODE>(0)));
    // Transpose dest step 1 (FPU)
    MATH((llk_math_deepseek_moe_gate_transpose_dest_single_face_step1_init<is_32bit>()));
    MATH((llk_math_deepseek_moe_gate_transpose_dest_single_face_step1<DST_ACCUM_MODE, is_32bit>()));
    // Top8 (SFPU)
    MATH((llk_math_sfpu_deepseek_moe_gate_top8<APPROX, DST_ACCUM_MODE>(0, eps, scale)));
    // Transpose dest step 2 (FPU)
    MATH((llk_math_deepseek_moe_gate_transpose_dest_single_face_step2_init<is_32bit>()));
    MATH((llk_math_deepseek_moe_gate_transpose_dest_single_face_step2<DST_ACCUM_MODE, is_32bit>()));
}

}  // namespace ckernel
