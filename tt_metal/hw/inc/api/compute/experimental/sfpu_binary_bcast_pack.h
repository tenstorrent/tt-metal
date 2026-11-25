// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"

#if defined(TRISC_PACK) && !defined(ARCH_QUASAR)
#include "ckernel_sfpu_binary_bcast.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif

namespace ckernel {

#ifndef ARCH_QUASAR
// clang-format off
/**
 * Pack-thread (TRISC2) variants of `sfpu_add_bcast_row_init` / `sfpu_add_bcast_row`
 * from api/compute/sfpu_binary_bcast.h, for a kernel whose fused activation also
 * runs on the pack thread (the SFPU registers are shared, so both must run on the
 * same thread). Same semantics and preconditions as the math-thread versions.
 * BH/WH only.
 */
// clang-format on
ALWI void sfpu_add_bcast_row_init_pack() {
    PACK((SFPU_BINARY_INIT_FN(unused, sfpu::_sfpu_binary_bcast_init_, (ckernel::BroadcastType::ROW))));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sfpu_add_bcast_row_pack(uint32_t dst_data_idx, uint32_t dst_row_vec_idx) {
    PACK((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        _calculate_sfpu_binary_bcast_full_tile_,
        (ckernel::BinaryOp::ADD, ckernel::BroadcastType::ROW),
        dst_data_idx,
        dst_row_vec_idx,
        dst_data_idx,
        VectorMode::None)));
}
#endif

}  // namespace ckernel
