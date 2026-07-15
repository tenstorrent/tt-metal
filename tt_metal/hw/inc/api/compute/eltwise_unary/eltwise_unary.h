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

// Legacy combined hardware + pipeline init for eltwise-unary / SFPU ops. The current programming
// model is: compute_kernel_hw_startup(icb, ocb) once at the start of MAIN, then copy_init(icb).
//
// NOTE: the body is kept verbatim (full one-time HW config + the dual SrcA/Pack state_configure)
// instead of forwarding to compute_kernel_hw_startup + copy_init. Old callers invoke this mid-kernel
// as an op-to-op reconfig and rely on the full HW configuration; compute_kernel_hw_startup is unsafe
// mid-kernel and records no reconfig diff. Keeping the body verbatim preserves both the SrcA+Pack
// reconfig diff the compute-kernel sentinel asserts and the HW config old callers depend on.
[[deprecated(
    "Use compute_kernel_hw_startup(icb, ocb) once at kernel start, then copy_init(icb). This will be removed after "
    "15-09-2026.")]] ALWI void
unary_op_init_common(uint32_t icb, uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
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

// Legacy SFPU init; forwards to unary_op_init_common. Superseded by
// compute_kernel_hw_startup(icb, ocb) + copy_init(icb).
[[deprecated(
    "Use compute_kernel_hw_startup(icb, ocb) once at kernel start, then copy_init(icb). This will be removed after "
    "15-09-2026.")]] ALWI void
init_sfpu(uint32_t icb, uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
    unary_op_init_common(icb, ocb, call_line);
}

}  // namespace ckernel
