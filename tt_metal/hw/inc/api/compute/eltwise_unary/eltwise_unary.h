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
    // Drain any pending coprocessor instructions before reconfiguring the packer.
    // Without this, a stale REG2FLOP left in the coprocessor queue by a prior
    // operation (e.g. tilize) can be processed by the tensix_sync() inside
    // llk_pack_dest_init *after* we have already written the correct pack config,
    // overwriting it with the stale value.  Flushing here ensures stale entries
    // complete before our correct config writes arrive.
    tensix_sync();

    state_configure<Operand::SRCA, Operand::PACK>(icb, ocb, call_line);

    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE, true>(icb)));
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        false /*transpose of faces*/, false /*transpose within 16x16 face*/, icb)));

    PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(ocb)));
    PACK((llk_pack_init<false>(ocb)));
    // Second drain on the PACK thread, after llk_pack_hw_configure/llk_pack_init
    // have queued the new pack config writes, but before llk_pack_dest_init's
    // internal tensix_sync runs.  The all-threads tensix_sync above drains stale
    // PACK REG2FLOPs from a prior op, but the all-threads sync also delays MATH
    // and UNPACK at startup; that perturbation reorders MATH/UNPACK cfg writes
    // versus PACK's, which on the binary-ng UnpackToDestEn (uint16) path opens
    // a multi-tile race that corrupts large 5D-broadcast outputs.  The extra
    // PACK sync here re-balances PACK against MATH/UNPACK timing without
    // weakening the original drain.
    PACK((tensix_sync()));
    PACK((llk_pack_dest_init<DST_ACCUM_MODE, false>()));

    MATH((llk_math_eltwise_unary_datacopy_init<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE>(icb)));
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
    unary_op_init_common(icb, ocb, call_line);
}

}  // namespace ckernel
