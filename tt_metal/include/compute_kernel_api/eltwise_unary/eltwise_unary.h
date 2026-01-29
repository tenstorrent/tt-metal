// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/sentinel/compute_kernel_sentinel.h"
#ifdef TRISC_MATH
#include "llk_math_unary_datacopy_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_A_api.h"
#include "llk_unpack_common_api.h"
#endif

namespace ckernel {

ALWI void unary_op_init_common(uint32_t icb, uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
    state_configure<Operand::SRCA, Operand::PACK>(icb, ocb, call_line);

#ifdef TRISC_UNPACK

    // 32bit formats are implemented using unpack to dest, since SrcB is only 19bits wide
    const std::uint32_t dst_format = get_operand_dst_format(icb);
    const bool enable_unpack_to_dest = (dst_format == (std::uint32_t)DataFormat::Float32) ||
                                       (dst_format == (std::uint32_t)DataFormat::UInt32) ||
                                       (dst_format == (std::uint32_t)DataFormat::Int32);

    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE, true>(icb)));
    if (enable_unpack_to_dest) {
        UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
            false /*transpose of faces*/, false /*transpose within 16x16 face*/, icb)));
    } else {
        UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
            false /*transpose of faces*/, false /*transpose within 16x16 face*/, icb)));
    }
#endif
    PACK((llk_pack_hw_configure_disaggregated<DST_ACCUM_MODE, false>(ocb)));
    PACK((llk_pack_init<false>(ocb)));
    PACK((llk_pack_dest_init<DST_ACCUM_MODE, false>()));

    MATH((llk_math_eltwise_unary_datacopy_init<A2D, DST_ACCUM_MODE, BroadcastType::NONE>(icb)));
    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(icb, icb)));
}

ALWI void init_sfpu(uint32_t icb, uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
    unary_op_init_common(icb, ocb, call_line);
}

}  // namespace ckernel
