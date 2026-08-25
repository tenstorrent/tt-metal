// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"
#include "api/compute/sentinel/compute_kernel_sentinel.h"
#ifdef TRISC_MATH
#include "llk_math_unary_datacopy_api.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_A_api.h"
#endif

namespace ckernel {

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void unary_op_init_common(uint32_t icb, uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
#ifndef ARCH_QUASAR
    state_configure<Operand::SRCA, Operand::PACK>(icb, ocb, call_line);

    UNPACK((llk_unpack_hw_configure<is_fp32_dest_acc_en>(icb)));
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        false /*transpose of faces*/, false /*transpose within 16x16 face*/, icb)));

    PACK((llk_pack_hw_configure<is_fp32_dest_acc_en>(ocb)));
    PACK((llk_pack_init(ocb)));
    PACK((llk_pack_dest_init<is_fp32_dest_acc_en, PackMode::Default>(ocb)));

    MATH((llk_math_eltwise_unary_datacopy_init<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE>(icb)));
    MATH((llk_math_pack_sync_init<is_fp32_dest_acc_en>()));
    MATH((llk_math_hw_configure<is_fp32_dest_acc_en>(icb, icb)));
    // Eltwise unary / SFPU ops keep the Src zero-substitution flag disabled to preserve bf16 -0.0.
    // Asserted after hw_configure (which sets the operand-driven DEFAULT) so it is the last writer
    // before the op runs; skip-if-set keeps it cheap.
    MATH((ckernel::math::_configure_unary_preserve_zero_flag_state_()));
#else
    UNPACK((llk_unpack_hw_configure(icb)));
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        0 /*transpose of faces*/, 0 /*transpose within 16x16 face*/, icb)));

    PACK((llk_pack_hw_configure<is_fp32_dest_acc_en>(ocb)));
    PACK((llk_pack_init(ocb)));
    PACK((llk_pack_dest_init()));

    MATH((llk_math_eltwise_unary_datacopy_init<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, UnpackToDestEn>(
        icb)));
    MATH((llk_math_pack_sync_init()));
    MATH((llk_math_hw_configure<is_fp32_dest_acc_en>(icb, icb)));
#endif
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void init_sfpu(uint32_t icb, uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
    unary_op_init_common<is_fp32_dest_acc_en>(icb, ocb, call_line);
}

}  // namespace ckernel
