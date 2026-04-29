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

ALWI void unary_op_init_common(uint32_t icb, uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
#ifndef ARCH_QUASAR
    state_configure<Operand::SRCA, Operand::PACK>(icb, ocb, call_line);

    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE, true>(icb)));
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        false /*transpose of faces*/, false /*transpose within 16x16 face*/, icb)));

    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(ocb)));
    PACK((llk_pack_init<false>(ocb)));
    // Drain any pending PACK-thread coprocessor instructions before
    // llk_pack_dest_init's internal tensix_sync runs. Otherwise a stale REG2FLOP
    // left in the PACK queue by a prior operation (e.g. tilize) can be processed
    // *after* we have already written the correct pack config, overwriting it.
    // Scoped to PACK only — UNPACK and MATH have their own per-thread coproc
    // queues and adding the sync to the all-threads prefix perturbs cfg-write
    // ordering on those threads (race window seen on multi-tile uint16 broadcast).
    PACK((tensix_sync()));
    PACK((llk_pack_dest_init<DST_ACCUM_MODE, false>()));

    MATH((llk_math_eltwise_unary_datacopy_init<A2D, DST_ACCUM_MODE, BroadcastType::NONE>(icb)));
    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(icb, icb)));
#else
    UNPACK((llk_unpack_hw_configure(icb)));
    UNPACK((llk_unpack_A_init<false /*transpose*/, DST_ACCUM_MODE>(icb)));

    PACK((llk_pack_hw_configure(ocb)));
    PACK((llk_pack_init(ocb)));

    MATH((llk_math_eltwise_unary_datacopy_init<ckernel::DataCopyType::A2D, DST_ACCUM_MODE>(icb)));
    MATH((llk_math_pack_sync_init()));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(icb, icb)));
#endif
}

ALWI void init_sfpu(uint32_t icb, uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
#ifndef ARCH_QUASAR
    unary_op_init_common(icb, ocb, call_line);
#endif  // TODO: AM; add Quasar implementation
}

}  // namespace ckernel
