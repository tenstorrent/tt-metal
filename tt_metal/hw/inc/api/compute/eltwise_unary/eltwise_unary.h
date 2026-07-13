// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"
#include "api/compute/sentinel/compute_kernel_sentinel.h"
#ifdef TRISC_MATH
#include "llk_math_unary_datacopy_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_A_api.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs the common hardware and op init for eltwise-unary / SFPU kernels: configures the
 * unpacker, the packer, and the math unit for an A2D datacopy pipeline from the input CB to the
 * output CB. Call once before issuing eltwise-unary / SFPU tile ops.
 *
 * Note: the hardware-configuration portion of this init is a candidate to move into
 * compute_kernel_hw_startup in a future pass, leaving this as a slimmer op-only init.
 *
 * Return value: None
 *
 * | Argument | Description                                       | Type     | Valid Range | Required |
 * |----------|---------------------------------------------------|----------|-------------|----------|
 * | icb      | The identifier of the input circular buffer (CB)  | uint32_t | 0 to 31     | True     |
 * | ocb      | The identifier of the output circular buffer (CB) | uint32_t | 0 to 31     | True     |
 */
// clang-format on
ALWI void unary_op_init_common(uint32_t icb, uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
#ifndef ARCH_QUASAR
    state_configure<Operand::SRCA, Operand::PACK>(icb, ocb, call_line);

    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(icb)));
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        false /*transpose of faces*/, false /*transpose within 16x16 face*/, icb)));

    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(ocb)));
    PACK((llk_pack_init(ocb)));
    PACK((llk_pack_dest_init<DST_ACCUM_MODE, PackMode::Default>()));

    MATH((llk_math_eltwise_unary_datacopy_init<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE>(icb)));
    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(icb, icb)));
    // Eltwise unary / SFPU ops keep the Src zero-substitution flag disabled to preserve bf16 -0.0.
    // Asserted after hw_configure (which sets the operand-driven DEFAULT) so it is the last writer
    // before the op runs; skip-if-set keeps it cheap.
    MATH((ckernel::math::_configure_unary_preserve_zero_flag_state_()));
#else
    UNPACK((llk_unpack_hw_configure(icb)));
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        0 /*transpose of faces*/, 0 /*transpose within 16x16 face*/, icb)));

    PACK((llk_pack_hw_configure(ocb)));
    PACK((llk_pack_init(ocb)));
    PACK((llk_pack_dest_init()));

    MATH((llk_math_eltwise_unary_datacopy_init<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
        icb)));
    MATH((llk_math_pack_sync_init()));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(icb, icb)));
#endif
}

// clang-format off
/**
 * Alias of unary_op_init_common, kept for SFPU-kernel readability. Performs the common hardware
 * and op init for SFPU kernels; see unary_op_init_common for details.
 * Return value: None
 *
 * | Argument | Description                                       | Type     | Valid Range | Required |
 * |----------|---------------------------------------------------|----------|-------------|----------|
 * | icb      | The identifier of the input circular buffer (CB)  | uint32_t | 0 to 31     | True     |
 * | ocb      | The identifier of the output circular buffer (CB) | uint32_t | 0 to 31     | True     |
 */
// clang-format on
ALWI void init_sfpu(uint32_t icb, uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
    unary_op_init_common(icb, ocb, call_line);
}

}  // namespace ckernel
